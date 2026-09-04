import cv2
import json
import os
import shutil
import struct
from pathlib import Path

INPUT_VIDEO = "video.mp4"
OUTPUT_ROOT = "chunks"

MAX_SECONDS = 45
FPS = 10

# عدد الإطارات داخل كل ملف.
# 5 مناسب كبداية: يقلل عدد HTTP requests بدون جعل الملف ضخمًا جدًا.
FRAMES_PER_CHUNK = 5

PROFILES = {
    "720p": {
        "width": 1280,
        "height": 720,
    },
    "480p": {
        "width": 854,
        "height": 480,
    },
    "360p": {
        "width": 640,
        "height": 360,
    },
}


def rgb_to_rgb565(r, g, b):
    """
    RGB888 -> RGB565
    16 bits لكل بكسل.
    """

    r5 = (int(r) * 31 + 127) // 255
    g6 = (int(g) * 63 + 127) // 255
    b5 = (int(b) * 31 + 127) // 255

    return (r5 << 11) | (g6 << 5) | b5


def encode_frame_rgb565(frame):
    """
    OpenCV BGR -> RGB565 binary.

    النتيجة:
        2 bytes لكل pixel
    """

    height, width = frame.shape[:2]

    output = bytearray(width * height * 2)
    offset = 0

    for y in range(height):
        row = frame[y]

        for x in range(width):
            b, g, r = row[x]

            value = rgb_to_rgb565(r, g, b)

            # little-endian
            output[offset] = value & 0xFF
            output[offset + 1] = (value >> 8) & 0xFF

            offset += 2

    return bytes(output)


def write_chunk(path, frames):
    """
    نخزن frames متتالية بدون JSON.
    """
    with open(path, "wb") as f:
        for frame_data in frames:
            f.write(frame_data)


def process_profile(cap, profile_name, width, height):
    print(f"\n=== Processing {profile_name}: {width}x{height} ===")

    output_dir = Path(OUTPUT_ROOT) / profile_name

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # إعادة فتح الفيديو لكل profile
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    max_frames = int(FPS * MAX_SECONDS)

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if not source_fps or source_fps <= 0:
        source_fps = FPS

    # Sampling مناسب للوصول إلى FPS المطلوب
    frame_step = source_fps / FPS

    next_source_frame = 0.0
    processed = 0
    source_index = 0

    chunk_frames = []
    chunk_index = 0

    while processed < max_frames:

        # تخطي frames حتى نصل للـ frame المطلوب
        while source_index < int(next_source_frame):
            ok = cap.grab()

            if not ok:
                break

            source_index += 1

        ok, frame = cap.read()

        if not ok:
            break

        source_index += 1

        resized = cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

        encoded = encode_frame_rgb565(resized)

        chunk_frames.append(encoded)

        processed += 1
        next_source_frame += frame_step

        if len(chunk_frames) >= FRAMES_PER_CHUNK:
            filename = output_dir / f"chunk_{chunk_index:04d}.bin"

            write_chunk(filename, chunk_frames)

            chunk_frames = []
            chunk_index += 1

            print(
                f"{profile_name}: "
                f"{processed}/{max_frames} frames"
            )

    # آخر chunk
    if chunk_frames:
        filename = output_dir / f"chunk_{chunk_index:04d}.bin"
        write_chunk(filename, chunk_frames)
        chunk_index += 1

    manifest = {
        "profile": profile_name,
        "width": width,
        "height": height,
        "fps": FPS,
        "total_frames": processed,
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "chunks": chunk_index,
        "format": "RGB565_LE",
        "bytes_per_pixel": 2,
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Done {profile_name}: "
        f"{processed} frames / {chunk_index} chunks"
    )


def main():
    if not os.path.exists(INPUT_VIDEO):
        print(f"ERROR: {INPUT_VIDEO} not found")
        return

    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    os.makedirs(OUTPUT_ROOT)

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    for profile_name, profile in PROFILES.items():
        process_profile(
            cap,
            profile_name,
            profile["width"],
            profile["height"]
        )

    cap.release()

    master_manifest = {
        "source": INPUT_VIDEO,
        "duration_limit": MAX_SECONDS,
        "fps": FPS,
        "profiles": list(PROFILES.keys()),
    }

    with open(
        Path(OUTPUT_ROOT) / "manifest.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(master_manifest, f, indent=2)

    print("\n================================")
    print("VIDEO PROCESSING COMPLETE")
    print("Profiles: 720p / 480p / 360p")
    print("================================")


if __name__ == "__main__":
    main()
