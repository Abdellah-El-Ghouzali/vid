import cv2
import json
import math
import os
import re
import struct
import sys
import numpy as np
import zstandard as zstd

# ============================================================
# CONFIGURATION
# ============================================================
VERSION = 5  # Version 5: 2.0s segments (cuts HTTP requests in half)
FPS = 15
MAX_SECONDS = None  # None = full video duration

PROFILES = {
    "288p": (288, 512)
}

FIRST_SEGMENT_FRAMES = 5   # ~0.33s ultralight startup segment (< 150 KB)
SEGMENT_FRAMES = 30        # 2.0s segments (halves HTTP calls, doubles buffer runway)
MAX_SEGMENT_BYTES = 1_500_000
TILE_SIZE = 8
DELTA_MAX_RATIO = 0.85
ZSTD_LEVEL = 9             # Maximum compression for fast downloads

# Precomputed Lookup Tables for instant RGB565 conversion
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
    return (LUT_R[frame[:, :, 2]] | LUT_G[frame[:, :, 1]] | LUT_B[frame[:, :, 0]]).astype("<u2")


def write_u32(file, value):
    file.write(struct.pack("<I", int(value)))


# ============================================================
# VECTORIZED NOISE-FILTERED DELTA PAYLOAD
# ============================================================

def build_delta_payload(prev_tiles, curr_tiles, tiles_y, tiles_x, tile_size):
    r_prev = (prev_tiles >> 11) & 0x1F
    g_prev = (prev_tiles >> 5) & 0x3F
    b_prev = prev_tiles & 0x1F

    r_curr = (curr_tiles >> 11) & 0x1F
    g_curr = (curr_tiles >> 5) & 0x3F
    b_curr = curr_tiles & 0x1F

    diff_r = np.abs(r_curr.astype(np.int16) - r_prev.astype(np.int16))
    diff_g = np.abs(g_curr.astype(np.int16) - g_prev.astype(np.int16))
    diff_b = np.abs(b_curr.astype(np.int16) - b_prev.astype(np.int16))

    sig_pixel = (diff_r > 1) | (diff_g > 2) | (diff_b > 1)
    diff_mask = np.count_nonzero(sig_pixel, axis=(2, 3)) >= 2
    ys, xs = np.nonzero(diff_mask)
    count = len(ys)

    if count == 0:
        return struct.pack("<H", 0), 0

    sel_tiles = curr_tiles[ys, xs]

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


def generate_bundle(profile_dir, manifest_path):
    with open(manifest_path, "rb") as mf:
        manifest_bytes = mf.read()

    manifest_data = json.loads(manifest_bytes.decode("utf-8"))
    segments = manifest_data.get("segments", [])
    if not segments:
        return None

    bundle_path = os.path.join(profile_dir, "bundle.bin")
    with open(bundle_path, "wb") as bf:
        write_u32(bf, len(manifest_bytes))
        bf.write(manifest_bytes)
        write_u32(bf, len(segments))

        for seg in segments:
            seg_file = os.path.join(profile_dir, seg["filename"])
            seg_size = os.path.getsize(seg_file)
            write_u32(bf, seg_size)
            with open(seg_file, "rb") as sf:
                bf.write(sf.read())

    total_size = os.path.getsize(bundle_path)
    return bundle_path, total_size


# ============================================================
# PROFILE PROCESSOR
# ============================================================

def process_profile(video_path, output_dir, profile_name, width, height):
    profile_dir = os.path.join(output_dir, profile_name)
    os.makedirs(profile_dir, exist_ok=True)

    print(f"  --> Encoding Profile: {profile_name} ({width}x{height} @ {FPS}fps)")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_duration = source_frame_count / source_fps if source_frame_count > 0 else 0
    duration = source_duration if MAX_SECONDS is None else min(source_duration, MAX_SECONDS)
    expected_total_frames = max(1, int(math.floor(duration * FPS)))

    compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    frames = []
    sample_step = source_fps / FPS
    next_sample_source_frame = 0.0
    source_index = 0

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
        raise RuntimeError(f"No frames extracted from {video_path}")

    tiles_y = math.ceil(height / TILE_SIZE)
    tiles_x = math.ceil(width / TILE_SIZE)
    total_tiles = tiles_y * tiles_x
    max_delta_uncompressed_bytes = int(width * height * 2 * DELTA_MAX_RATIO)

    segments = []
    frame_index = 0
    segment_index = 0

    reshaped_frames = [
        f.reshape(tiles_y, TILE_SIZE, tiles_x, TILE_SIZE).swapaxes(1, 2)
        for f in frames
    ]

    while frame_index < total_frames:
        target_frames = FIRST_SEGMENT_FRAMES if segment_index == 0 else SEGMENT_FRAMES
        segment_start = frame_index

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

            if changed_count * (2 + TILE_SIZE * TILE_SIZE * 2) >= max_delta_uncompressed_bytes:
                frame_type = 0
                frame_data = compressor.compress(curr_frame.tobytes())
            else:
                delta_compressed = compressor.compress(delta_payload)
                # Keep delta active across camera movement up to 60% tile changes
                if changed_count < total_tiles * 0.60:
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
        segment_index += 1

    manifest = {
        "version": VERSION,
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

    bundle_res = generate_bundle(profile_dir, manifest_path)
    if bundle_res:
        _, b_size = bundle_res
        print(f"     -> Generated bundle.bin: {b_size / (1024 * 1024):.2f} MB")

    print(f"     -> Finished: {total_frames} frames ({manifest['duration']:.1f}s) in {len(segments)} segments.")
    return manifest


def parse_video_id(name: str):
    match = re.search(r'\d+', name)
    return int(match.group(0)) if match else None


def discover_videos(base_dir="videos"):
    video_tasks = []
    if not os.path.exists(base_dir):
        return video_tasks

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            vid_id = parse_video_id(item)
            if vid_id is not None:
                video_files = [f for f in os.listdir(item_path) if f.lower().endswith(('.mp4', '.mov', '.mkv', '.webm'))]
                if video_files:
                    source_file = os.path.join(item_path, video_files[0])
                    out_dir = os.path.join(item_path, "chunks")
                    video_tasks.append((vid_id, source_file, out_dir))
        elif os.path.isfile(item_path) and item.lower().endswith(('.mp4', '.mov', '.mkv', '.webm')):
            vid_id = parse_video_id(item)
            if vid_id is not None:
                out_dir = os.path.join(base_dir, str(vid_id), "chunks")
                video_tasks.append((vid_id, item_path, out_dir))

    video_tasks.sort(key=lambda x: x[0])
    return video_tasks


def should_skip(video_path, output_dir):
    for profile_name in PROFILES.keys():
        profile_dir = os.path.join(output_dir, profile_name)
        manifest_path = os.path.join(profile_dir, "manifest.json")
        bundle_path = os.path.join(profile_dir, "bundle.bin")

        if not os.path.exists(manifest_path) or not os.path.exists(bundle_path):
            return False

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version", 0) < VERSION:
                return False
        except Exception:
            return False

        if os.path.getmtime(manifest_path) < os.path.getmtime(video_path):
            return False

    return True


def main():
    force_rebuild = "--force" in sys.argv
    tasks = discover_videos("videos")

    if not tasks and os.path.isfile("video.mp4"):
        tasks = [(1, "video.mp4", "videos/1/chunks")]

    if not tasks:
        print("No videos found to process.")
        return

    print("=" * 75)
    print(f"FOUND {len(tasks)} VIDEO(S) TO PROCESS (ENCODER v{VERSION})")
    print("=" * 75)

    for vid_id, video_path, out_dir in tasks:
        print(f"\n[VIDEO #{vid_id}] Source: {video_path}")

        if not force_rebuild and should_skip(video_path, out_dir):
            print(f"  --> [SKIPPED] Up-to-date with Version {VERSION} compression.")
            continue

        manifests = {}
        for profile_name, (w, h) in PROFILES.items():
            manifests[profile_name] = process_profile(video_path, out_dir, profile_name, w, h)

        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifests, f, indent=2)

    print("\n" + "=" * 75)
    print("ALL VIDEOS SUCCESSFULLY PROCESSED AND COMPRESSED!")
    print("=" * 75)


if __name__ == "__main__":
    main()
