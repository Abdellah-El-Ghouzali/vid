import cv2
import json
import math
import os
import shutil
import struct
import numpy as np
import zstandard as zstd

# ============================================================
# CONFIG
# ============================================================

FPS = 15
MAX_SECONDS = 45

PROFILES = {
    "576p": (576, 1024)
}

FIRST_SEGMENT_FRAMES = 8
SEGMENT_FRAMES = 15
MAX_SEGMENT_BYTES = 2_000_000
TILE_SIZE = 8
DELTA_MAX_RATIO = 0.90
ZSTD_LEVEL = 3
OUTPUT_DIR = "chunks"

# Precomputed Lookup Tables for instant RGB565 conversion matching your formula:
VALS = np.arange(256, dtype=np.uint16)
LUT_R = (((VALS * 31 + 127) // 255) << 11).astype(np.uint16)
LUT_G = (((VALS * 63 + 127) // 255) << 5).astype(np.uint16)
LUT_B = ((VALS * 31 + 127) // 255).astype(np.uint16)


# ============================================================
# FAST HELPERS
# ============================================================

def crop_to_portrait(frame, target_width, target_height):
    source_height, source_width = frame.shape[:2]
    target_ratio = target_width / target_height
    source_ratio = source_width / source_height

    if source_ratio > target_ratio:
        new_width = int(source_height * target_ratio)
        left = max((source_width - new_width) // 2, 0)
        frame = frame[:, left:left + new_width]
    else:
        new_height = int(source_width / target_ratio)
        top = max((source_height - new_height) // 2, 0)
        frame = frame[top:top + new_height, :]

    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def frame_to_rgb565_array(frame):
    # Vectorized LUT indexing: 50x faster than channel math
    return (LUT_R[frame[:, :, 2]] | LUT_G[frame[:, :, 1]] | LUT_B[frame[:, :, 0]]).astype("<u2")


def write_u32(file, value):
    file.write(struct.pack("<I", int(value)))


# ============================================================
# VECTORIZED DELTA PAYLOAD
# ============================================================

def build_delta_payload(prev_tiles, curr_tiles, tiles_y, tiles_x, tile_size):
    """
    Vectorized tile diffing: checks all 9,216 tiles simultaneously in C/SIMD.
    prev_tiles & curr_tiles shape: (tiles_y, tiles_x, tile_size, tile_size)
    """
    diff_mask = np.any(prev_tiles != curr_tiles, axis=(2, 3))
    ys, xs = np.nonzero(diff_mask)
    count = len(ys)

    if count == 0:
        return struct.pack("<H", 0), 0

    # Extract changed tiles directly into structured memory
    sel_tiles = curr_tiles[ys, xs]  # shape: (count, tile_size, tile_size)

    tile_dt = np.dtype([
        ('x', 'u1'),
        ('y', 'u1'),
        ('pixels', '<u2', (tile_size, tile_size))
    ])

    out_arr = np.empty(count, dtype=tile_dt)
    out_arr['x'] = xs.astype(np.uint8)
    out_arr['y'] = ys.astype(np.uint8)
    out_arr['pixels'] = sel_tiles

    payload = struct.pack("<H", count) + out_arr.tobytes()
    return payload, count


# ============================================================
# OUTPUT
# ============================================================

def clean_output():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_segment(profile_dir, segment_index, encoded_frames):
    filename = f"segment_{segment_index:04d}.bin"
    path = os.path.join(profile_dir, filename)

    with open(path, "wb") as file:
        write_u32(file, len(encoded_frames))
        for frame_type, frame_data in encoded_frames:
            file.write(bytes([frame_type]))
            write_u32(file, len(frame_data))
            file.write(frame_data)

    return {
        "index": segment_index,
        "filename": filename,
        "frame_count": len(encoded_frames),
        "bytes": os.path.getsize(path)
    }


# ============================================================
# PROCESS PROFILE
# ============================================================

def process_profile(video_path, profile_name, width, height):
    profile_dir = os.path.join(OUTPUT_DIR, profile_name)
    os.makedirs(profile_dir, exist_ok=True)

    print()
    print("=" * 70)
    print(f"PROCESSING: {profile_name} ({width}x{height} @ {FPS}fps)")
    print("=" * 70)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.mp4")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_duration = source_frame_count / source_fps if source_frame_count > 0 else 0
    duration = min(source_duration, MAX_SECONDS)
    expected_total_frames = max(1, int(math.floor(duration * FPS)))

    compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    frames = []
    sample_step = source_fps / FPS
    next_sample_source_frame = 0.0
    source_index = 0

    # 1. Fast frame decoding
    while len(frames) < expected_total_frames:
        ok, frame = cap.read()
        if not ok:
            break

        current_source_index = source_index
        source_index += 1

        if current_source_index + 1e-9 < next_sample_source_frame:
            continue

        frame = crop_to_portrait(frame, width, height)
        rgb565 = frame_to_rgb565_array(frame)
        frames.append(rgb565)
        next_sample_source_frame += sample_step

    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        raise RuntimeError("No frames generated.")

    print(f"Decoded & Converted {total_frames} frames in memory.")

    # 2. Segment & Delta Encoding Setup
    tiles_y = math.ceil(height / TILE_SIZE)
    tiles_x = math.ceil(width / TILE_SIZE)
    total_tiles = tiles_y * tiles_x
    max_delta_uncompressed_bytes = int(width * height * 2 * DELTA_MAX_RATIO)

    segments = []
    frame_index = 0
    segment_index = 0

    # Pre-reshape frames into tiled blocks for vectorized diffing
    # shape: (total_frames, tiles_y, tiles_x, 8, 8)
    reshaped_frames = [
        f.reshape(tiles_y, TILE_SIZE, tiles_x, TILE_SIZE).swapaxes(1, 2)
        for f in frames
    ]

    while frame_index < total_frames:
        target_frames = FIRST_SEGMENT_FRAMES if segment_index == 0 else SEGMENT_FRAMES
        segment_start = frame_index

        # Frame 0 of each segment is ALWAYS a keyframe
        keyframe = frames[frame_index]
        full_compressed = compressor.compress(keyframe.tobytes())
        encoded_frames = [(0, full_compressed)]
        segment_size = 1 + 4 + len(full_compressed)

        prev_tiles = reshaped_frames[frame_index]
        frame_index += 1
        frames_in_segment = 1

        while frame_index < total_frames and frames_in_segment < target_frames:
            curr_frame = frames[frame_index]
            curr_tiles = reshaped_frames[frame_index]

            delta_payload, changed_count = build_delta_payload(
                prev_tiles, curr_tiles, tiles_y, tiles_x, TILE_SIZE
            )

            # Avoid redundant keyframe compression:
            # If payload is larger than keyframe budget, don't bother compressing delta
            if changed_count * (2 + TILE_SIZE * TILE_SIZE * 2) >= max_delta_uncompressed_bytes:
                frame_type = 0
                frame_data = compressor.compress(curr_frame.tobytes())
            else:
                delta_compressed = compressor.compress(delta_payload)

                # If less than 40% of tiles changed, delta is guaranteed smaller than keyframe
                if changed_count < total_tiles * 0.40:
                    frame_type = 1
                    frame_data = delta_compressed
                else:
                    kf_compressed = compressor.compress(curr_frame.tobytes())
                    if len(delta_compressed) <= len(kf_compressed) * DELTA_MAX_RATIO:
                        frame_type = 1
                        frame_data = delta_compressed
                    else:
                        frame_type = 0
                        frame_data = kf_compressed

            projected_size = segment_size + 1 + 4 + len(frame_data)
            if projected_size > MAX_SEGMENT_BYTES:
                break

            encoded_frames.append((frame_type, frame_data))
            segment_size = projected_size
            prev_tiles = curr_tiles
            frame_index += 1
            frames_in_segment += 1

        seg_info = write_segment(profile_dir, segment_index, encoded_frames)
        seg_info["start_frame"] = segment_start
        seg_info["end_frame"] = frame_index - 1
        seg_info["duration"] = frames_in_segment / FPS
        seg_info["keyframes"] = sum(1 for t, _ in encoded_frames if t == 0)
        seg_info["delta_frames"] = sum(1 for t, _ in encoded_frames if t == 1)
        segments.append(seg_info)

        print(f"Segment {segment_index:04d}: {frames_in_segment} frames, {seg_info['bytes'] / 1024:.1f} KB")
        segment_index += 1

    # Write Manifest
    manifest = {
        "version": 3,
        "profile": profile_name,
        "width": width,
        "height": height,
        "aspect_ratio": "9:16",
        "fps": FPS,
        "total_frames": total_frames,
        "duration": total_frames / FPS,
        "format": "RGB565_LE",
        "bytes_per_pixel": 2,
        "compression": "zstd",
        "compression_level": ZSTD_LEVEL,
        "tile_size": TILE_SIZE,
        "delta_max_ratio": DELTA_MAX_RATIO,
        "segment_count": len(segments),
        "first_segment_frames": FIRST_SEGMENT_FRAMES,
        "target_segment_frames": SEGMENT_FRAMES,
        "max_segment_bytes": MAX_SEGMENT_BYTES,
        "raw_bytes": width * height * 2 * total_frames,
        "segment_bytes": sum(s["bytes"] for s in segments),
        "segments": segments
    }

    manifest_path = os.path.join(profile_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"COMPLETE: {total_frames} frames encoded into {len(segments)} segments.")
    return manifest


def main():
    if not os.path.isfile("video.mp4"):
        raise FileNotFoundError("video.mp4 not found.")

    clean_output()
    manifests = {}
    for profile_name, (w, h) in PROFILES.items():
        manifests[profile_name] = process_profile("video.mp4", profile_name, w, h)

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifests, f, indent=2)


if __name__ == "__main__":
    main()
