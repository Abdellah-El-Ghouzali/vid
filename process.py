import cv2
import json
import shutil
from pathlib import Path

import numpy as np
import zstandard as zstd


# =========================================================
# CONFIG
# =========================================================

INPUT_VIDEO = "video.mp4"
OUTPUT_DIR = Path("chunks")

FPS = 10
MAX_SECONDS = 45

# كل ملف يحتوي عدة إطارات.
# 5 مناسب كبداية: عدد Requests أقل مع حجم ملف معقول.
FRAMES_PER_CHUNK = 5

# أعلى جودة عملية داخل EditableImage:
# 1280x720 غير ممكن كصورة EditableImage واحدة.
PROFILES = {
    "720p": (1024, 576),
    "480p": (854, 480),
    "360p": (640, 360),
}

ZSTD_LEVEL = 3


# =========================================================
# RGB888 -> RGB565
# =========================================================

def frame_to_rgb565(frame_bgr, width, height):
    """
    تحويل BGR888 إلى RGB565 بسرعة باستخدام NumPy.

    الناتج:
        2 bytes / pixel
    """

    resized = cv2.resize(
        frame_bgr,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    b = resized[:, :, 0].astype(np.uint16)
    g = resized[:, :, 1].astype(np.uint16)
    r = resized[:, :, 2].astype(np.uint16)

    r5 = (r * 31 + 127) // 255
    g6 = (g * 63 + 127) // 255
    b5 = (b * 31 + 127) // 255

    rgb565 = (
        (r5 << 11) |
        (g6 << 5) |
        b5
    )

    # Little endian
    return rgb565.astype("<u2", copy=False).tobytes()


# =========================================================
# CHUNK WRITER
# =========================================================

def write_chunk(path, frames, compressor):
    """
    دمج عدة frames ثم ضغطها بـ Zstandard.
    """

    raw = b"".join(frames)

    compressed = compressor.compress(raw)

    with open(path, "wb") as f:
        f.write(compressed)

    return len(raw), len(compressed)


# =========================================================
# MAIN
# =========================================================

def main():

    input_path = Path(INPUT_VIDEO)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Video not found: {INPUT_VIDEO}"
        )

    # حذف النتائج القديمة
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # إنشاء مجلدات الجودات
    for profile in PROFILES:
        (OUTPUT_DIR / profile).mkdir(
            parents=True,
            exist_ok=True
        )

    print("=" * 60)
    print("ROBLOX VIDEO ENCODER")
    print("=" * 60)

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if not source_fps or source_fps <= 0:
        source_fps = FPS

    source_frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    source_duration = (
        source_frame_count / source_fps
        if source_fps > 0
        else 0
    )

    max_frames = min(
        int(source_duration * FPS),
        FPS * MAX_SECONDS
    )

    print(f"Source FPS      : {source_fps:.2f}")
    print(f"Output FPS      : {FPS}")
    print(f"Source frames   : {source_frame_count}")
    print(f"Output frames   : {max_frames}")
    print(f"Duration limit  : {MAX_SECONDS}s")
    print()

    # Compressor واحد لكل profile
    compressors = {
        profile: zstd.ZstdCompressor(
            level=ZSTD_LEVEL
        )
        for profile in PROFILES
    }

    # Buffers
    chunk_buffers = {
        profile: []
        for profile in PROFILES
    }

    chunk_indices = {
        profile: 0
        for profile in PROFILES
    }

    total_raw = {
        profile: 0
        for profile in PROFILES
    }

    total_compressed = {
        profile: 0
        for profile in PROFILES
    }

    output_frame_index = 0

    # Sampling
    source_index = 0
    next_sample = 0.0
    sample_step = source_fps / FPS

    while output_frame_index < max_frames:

        ok, frame = cap.read()

        if not ok:
            break

        current_source_index = source_index

        source_index += 1

        # هل هذا frame مطلوب؟
        if current_source_index + 0.0001 < next_sample:
            continue

        next_sample += sample_step

        # -----------------------------------------
        # Encode كل الجودات من نفس الـframe
        # -----------------------------------------

        for profile, (width, height) in PROFILES.items():

            encoded = frame_to_rgb565(
                frame,
                width,
                height
            )

            chunk_buffers[profile].append(encoded)

            if len(chunk_buffers[profile]) >= FRAMES_PER_CHUNK:

                chunk_id = chunk_indices[profile]

                output_path = (
                    OUTPUT_DIR /
                    profile /
                    f"chunk_{chunk_id:04d}.bin"
                )

                raw_size, compressed_size = write_chunk(
                    output_path,
                    chunk_buffers[profile],
                    compressors[profile]
                )

                total_raw[profile] += raw_size
                total_compressed[profile] += compressed_size

                chunk_buffers[profile].clear()
                chunk_indices[profile] += 1

        output_frame_index += 1

        if output_frame_index % 10 == 0:

            percent = (
                output_frame_index /
                max_frames *
                100
            )

            print(
                f"Frames: "
                f"{output_frame_index}/{max_frames} "
                f"({percent:.1f}%)"
            )

    cap.release()

    # -----------------------------------------
    # آخر chunks
    # -----------------------------------------

    for profile in PROFILES:

        if chunk_buffers[profile]:

            chunk_id = chunk_indices[profile]

            output_path = (
                OUTPUT_DIR /
                profile /
                f"chunk_{chunk_id:04d}.bin"
            )

            raw_size, compressed_size = write_chunk(
                output_path,
                chunk_buffers[profile],
                compressors[profile]
            )

            total_raw[profile] += raw_size
            total_compressed[profile] += compressed_size

            chunk_indices[profile] += 1

    # -----------------------------------------
    # Profile manifests
    # -----------------------------------------

    for profile, (width, height) in PROFILES.items():

        manifest = {
            "profile": profile,
            "width": width,
            "height": height,
            "fps": FPS,
            "total_frames": output_frame_index,
            "frames_per_chunk": FRAMES_PER_CHUNK,
            "chunks": chunk_indices[profile],

            "format": "RGB565_LE",
            "bytes_per_pixel": 2,

            "compression": "zstd",
            "compression_level": ZSTD_LEVEL,

            "raw_bytes": total_raw[profile],
            "compressed_bytes": total_compressed[profile],
        }

        with open(
            OUTPUT_DIR /
            profile /
            "manifest.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                manifest,
                f,
                indent=2
            )

    # -----------------------------------------
    # Master manifest
    # -----------------------------------------

    master = {
        "fps": FPS,
        "duration": min(
            MAX_SECONDS,
            output_frame_index / FPS
        ),
        "total_frames": output_frame_index,

        "profiles": {
            profile: {
                "width": width,
                "height": height,
            }
            for profile, (width, height)
            in PROFILES.items()
        }
    }

    with open(
        OUTPUT_DIR / "manifest.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            master,
            f,
            indent=2
        )

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)

    for profile in PROFILES:

        raw_mb = total_raw[profile] / 1024 / 1024
        compressed_mb = (
            total_compressed[profile] /
            1024 /
            1024
        )

        print(
            f"{profile:5s} | "
            f"raw: {raw_mb:8.1f} MB | "
            f"zstd: {compressed_mb:8.1f} MB | "
            f"chunks: {chunk_indices[profile]}"
        )


if __name__ == "__main__":
    main()
