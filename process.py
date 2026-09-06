import cv2
import json
import math
import os
import re
import shutil
import struct
import sys
import numpy as np
import zstandard as zstd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

VIDEOS_DIR = "videos"
OUTPUT_DIR = "chunks"

FPS = 15
MAX_SECONDS = 45

# 288p portrait resolution (optimal for 10-player lag-free mobile/PC)
WIDTH = 288
HEIGHT = 512
PROFILE_NAME = "288p"

FIRST_SEGMENT_FRAMES = 8
SEGMENT_FRAMES = 15
MAX_SEGMENT_BYTES = 2_000_000
TILE_SIZE = 8
DELTA_MAX_RATIO = 0.90
ZSTD_LEVEL = 3

# Precomputed LUT for instant RGB565 conversion
VALS = np.arange(256, dtype=np.uint16)
LUT_R = (((VALS * 31 + 127) // 255) << 11).astype(np.uint16)
LUT_G = (((VALS * 63 + 127) // 255) << 5).astype(np.uint16)
LUT_B = ((VALS * 31 + 127) // 255).astype(np.uint16)


def crop_to_portrait(frame, target_width, target_height):
    h, w = frame.shape[:2]
    target_ratio = target_width / target_height
    source_ratio = w / h

    if source_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = max((w - new_w) // 2, 0)
        frame = frame[:, left:left + new_w]
    else:
        new_h = int(w / target_ratio)
        top = max((h - new_h) // 2, 0)
        frame = frame[top:top + new_h, :]

    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def frame_to_rgb565(frame):
    return (LUT_R[frame[:, :, 2]] | LUT_G[frame[:, :, 1]] | LUT_B[frame[:, :, 0]]).astype("<u2")


def build_delta_payload(prev_tiles, curr_tiles, tiles_y, tiles_x, tile_size):
    diff_mask = np.any(prev_tiles != curr_tiles, axis=(2, 3))
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

    return struct.pack("<H", count) + out_arr.tobytes(), count


def write_segment(profile_dir, segment_index, encoded_frames):
    filename = f"segment_{segment_index:04d}.bin"
    path = os.path.join(profile_dir, filename)

    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(encoded_frames)))
        for ftype, fdata in encoded_frames:
            f.write(bytes([ftype]))
            f.write(struct.pack("<I", len(fdata)))
            f.write(fdata)

    return {
        "index": segment_index,
        "filename": filename,
        "frame_count": len(encoded_frames),
        "bytes": os.path.getsize(path)
    }


def process_video(video_path: Path, video_id: int, force: bool = False):
    target_dir = Path(OUTPUT_DIR) / str(video_id) / PROFILE_NAME
    manifest_file = target_dir / "manifest.json"

    # Skip if already processed and video has not been updated
    if not force and manifest_file.exists():
        if manifest_file.stat().st_mtime >= video_path.stat().st_mtime:
            print(f"[SKIP] Video {video_id} already processed.")
            return True

    print(f"\n[ENCODING] Video ID {video_id}: {video_path.name}")
    target_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        return False

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = min(source_frames / source_fps if source_frames > 0 else 0, MAX_SECONDS)
    expected_frames = max(1, int(math.floor(duration * FPS)))

    compressor = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    frames = []
    step = source_fps / FPS
    next_sample = 0.0
    src_idx = 0

    while len(frames) < expected_frames:
        ok, frame = cap.read()
        if not ok:
            break
        curr_idx = src_idx
        src_idx += 1
        if curr_idx + 1e-9 < next_sample:
            continue

        frame = crop_to_portrait(frame, WIDTH, HEIGHT)
        frames.append(frame_to_rgb565(frame))
        next_sample += step

    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        print(f"[ERROR] No frames extracted from {video_path}")
        return False

    tiles_y = math.ceil(HEIGHT / TILE_SIZE)
    tiles_x = math.ceil(WIDTH / TILE_SIZE)
    total_tiles = tiles_y * tiles_x
    max_delta_bytes = int(WIDTH * HEIGHT * 2 * DELTA_MAX_RATIO)

    reshaped_frames = [
        f.reshape(tiles_y, TILE_SIZE, tiles_x, TILE_SIZE).swapaxes(1, 2)
        for f in frames
    ]

    segments = []
    frame_idx = 0
    seg_idx = 0

    while frame_idx < total_frames:
        target_count = FIRST_SEGMENT_FRAMES if seg_idx == 0 else SEGMENT_FRAMES
        start_frame = frame_idx

        kf = frames[frame_idx]
        kf_comp = compressor.compress(kf.tobytes())
        encoded = [(0, kf_comp)]
        seg_size = 1 + 4 + len(kf_comp)

        prev_tiles = reshaped_frames[frame_idx]
        frame_idx += 1
        count = 1

        while frame_idx < total_frames and count < target_count:
            curr_frame = frames[frame_idx]
            curr_tiles = reshaped_frames[frame_idx]

            payload, changed_count = build_delta_payload(prev_tiles, curr_tiles, tiles_y, tiles_x, TILE_SIZE)

            if changed_count * (2 + TILE_SIZE * TILE_SIZE * 2) >= max_delta_bytes:
                ftype = 0
                fdata = compressor.compress(curr_frame.tobytes())
            else:
                d_comp = compressor.compress(payload)
                if changed_count < total_tiles * 0.40:
                    ftype = 1
                    fdata = d_comp
                else:
                    kf_comp = compressor.compress(curr_frame.tobytes())
                    if len(d_comp) <= len(kf_comp) * DELTA_MAX_RATIO:
                        ftype = 1
                        fdata = d_comp
                    else:
                        ftype = 0
                        fdata = kf_comp

            proj_size = seg_size + 1 + 4 + len(fdata)
            if proj_size > MAX_SEGMENT_BYTES:
                break

            encoded.append((ftype, fdata))
            seg_size = proj_size
            prev_tiles = curr_tiles
            frame_idx += 1
            count += 1

        info = write_segment(str(target_dir), seg_idx, encoded)
        info["start_frame"] = start_frame
        info["end_frame"] = frame_idx - 1
        info["duration"] = count / FPS
        segments.append(info)
        seg_idx += 1

    manifest = {
        "version": 3,
        "video_id": video_id,
        "profile": PROFILE_NAME,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "total_frames": total_frames,
        "duration": total_frames / FPS,
        "segment_count": len(segments),
        "segments": segments
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] Video {video_id} saved: {total_frames} frames in {len(segments)} segments.")
    return True


def main():
    force = "--force" in sys.argv
    videos_dir = Path(VIDEOS_DIR)
    chunks_dir = Path(OUTPUT_DIR)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if not videos_dir.exists():
        videos_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created '{VIDEOS_DIR}' folder. Place your 1.mp4, 2.mp4 files inside it.")
        return

    # Find all mp4 files
    video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mov"))
    if not video_files:
        print(f"No video files found in '{VIDEOS_DIR}/'.")
        return

    # Extract ID numbers from filenames (e.g., '1.mp4' -> 1, 'video_02.mp4' -> 2)
    processed_ids = []
    for vf in sorted(video_files):
        match = re.search(r'\d+', vf.stem)
        video_id = int(match.group()) if match else None
        if video_id is not None:
            ok = process_video(vf, video_id, force=force)
            if ok:
                processed_ids.append(video_id)

    # Generate master index.json for Roblox
    index_manifest = {
        "count": len(processed_ids),
        "video_ids": sorted(processed_ids)
    }
    with open(chunks_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index_manifest, f, indent=2)

    print(f"\n==================================================")
    print(f"ALL VIDEOS COMPLETE: {len(processed_ids)} videos ready in chunks/")
    print(f"==================================================")


if __name__ == "__main__":
    main()
