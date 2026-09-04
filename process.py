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

# Video FPS
FPS = 8

# أقصى مدة
MAX_SECONDS = 45

# Shorts 9:16
PROFILES = {
    "576p": (576, 1024),
}

# عدد الإطارات داخل كل chunk
FRAMES_PER_CHUNK = 2

# Zstandard compression
ZSTD_LEVEL = 3


# =========================================================
# CROP TO 9:16
# =========================================================

def crop_to_portrait(frame):
    """
    قص الإطار إلى نسبة 9:16 من المنتصف.
    """

    source_height, source_width = frame.shape[:2]

    target_ratio = 9 / 16
    source_ratio = source_width / source_height

    # الفيديو أعرض من 9:16
    if source_ratio > target_ratio:

        new_width = int(
            source_height * target_ratio
        )

        left = (
            source_width - new_width
        ) // 2

        frame = frame[
            :,
            left:left + new_width
        ]

    # الفيديو أطول من 9:16
    elif source_ratio < target_ratio:

        new_height = int(
            source_width / target_ratio
        )

        top = (
            source_height - new_height
        ) // 2

        frame = frame[
            top:top + new_height,
            :
        ]

    return frame


# =========================================================
# FRAME -> RGB565
# =========================================================

def frame_to_rgb565(
    frame_bgr,
    width,
    height
):
    """
    Crop + resize + BGR888 -> RGB565 LE
    """

    frame_bgr = crop_to_portrait(
        frame_bgr
    )

    resized = cv2.resize(
        frame_bgr,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    # BGR
    b = resized[:, :, 0].astype(
        np.uint16
    )

    g = resized[:, :, 1].astype(
        np.uint16
    )

    r = resized[:, :, 2].astype(
        np.uint16
    )

    # 8-bit -> RGB565
    r5 = (
        r * 31 + 127
    ) // 255

    g6 = (
        g * 63 + 127
    ) // 255

    b5 = (
        b * 31 + 127
    ) // 255

    rgb565 = (
        (r5 << 11)
        |
        (g6 << 5)
        |
        b5
    )

    return rgb565.astype(
        "<u2",
        copy=False
    ).tobytes()


# =========================================================
# WRITE CHUNK
# =========================================================

def write_chunk(
    path,
    frames,
    compressor
):
    """
    Combine frames and compress using Zstandard.
    """

    raw = b"".join(frames)

    compressed = compressor.compress(
        raw
    )

    path.write_bytes(
        compressed
    )

    return (
        len(raw),
        len(compressed)
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("ROBLOX PORTRAIT VIDEO ENCODER")
    print("=" * 60)

    # =====================================================
    # INPUT
    # =====================================================

    input_path = Path(
        INPUT_VIDEO
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Video not found: {INPUT_VIDEO}"
        )

    # =====================================================
    # CLEAN OUTPUT
    # =====================================================

    if OUTPUT_DIR.exists():

        print(
            "Removing old chunks..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # CREATE DIRECTORIES
    # =====================================================

    for profile in PROFILES:

        (
            OUTPUT_DIR / profile
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # =====================================================
    # OPEN VIDEO
    # =====================================================

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video"
        )

    # =====================================================
    # SOURCE INFO
    # =====================================================

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if (
        not source_fps
        or source_fps <= 0
    ):

        source_fps = FPS

    source_frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    source_duration = (
        source_frame_count
        / source_fps
    )

    # عدد الإطارات المطلوب
    max_frames = min(
        int(
            source_duration * FPS
        ),
        FPS * MAX_SECONDS
    )

    print(
        f"Source FPS : {source_fps:.2f}"
    )

    print(
        f"Output FPS : {FPS}"
    )

    print(
        f"Source frames : {source_frame_count}"
    )

    print(
        f"Source duration : "
        f"{source_duration:.2f}s"
    )

    print(
        f"Output frames : {max_frames}"
    )

    print(
        f"Max duration : "
        f"{MAX_SECONDS}s"
    )

    print()

    # =====================================================
    # COMPRESSORS
    # =====================================================

    compressors = {}

    for profile in PROFILES:

        compressors[profile] = (
            zstd.ZstdCompressor(
                level=ZSTD_LEVEL
            )
        )

    # =====================================================
    # BUFFERS
    # =====================================================

    buffers = {}

    for profile in PROFILES:

        buffers[profile] = []

    # =====================================================
    # COUNTERS
    # =====================================================

    chunk_indices = {}

    raw_totals = {}

    compressed_totals = {}

    for profile in PROFILES:

        chunk_indices[profile] = 0
        raw_totals[profile] = 0
        compressed_totals[profile] = 0

    # =====================================================
    # SAMPLING
    # =====================================================

    source_index = 0
    output_index = 0

    sample_step = (
        source_fps / FPS
    )

    next_sample = 0.0

    # =====================================================
    # PROCESS
    # =====================================================

    while output_index < max_frames:

        ok, frame = cap.read()

        if not ok:
            break

        current_source = source_index

        source_index += 1

        # -------------------------------------------------
        # Skip frames that are not sampled
        # -------------------------------------------------

        if (
            current_source + 0.0001
            < next_sample
        ):
            continue

        next_sample += sample_step

        # -------------------------------------------------
        # Encode all profiles
        # -------------------------------------------------

        for profile, size in PROFILES.items():

            width, height = size

            encoded = frame_to_rgb565(
                frame,
                width,
                height
            )

            buffers[profile].append(
                encoded
            )

            # -------------------------------------------------
            # Write complete chunk
            # -------------------------------------------------

            if (
                len(buffers[profile])
                >= FRAMES_PER_CHUNK
            ):

                chunk_id = (
                    chunk_indices[profile]
                )

                output_path = (
                    OUTPUT_DIR
                    / profile
                    / f"chunk_{chunk_id:04d}.bin"
                )

                (
                    raw_size,
                    compressed_size
                ) = write_chunk(
                    output_path,
                    buffers[profile],
                    compressors[profile]
                )

                raw_totals[profile] += (
                    raw_size
                )

                compressed_totals[profile] += (
                    compressed_size
                )

                buffers[profile].clear()

                chunk_indices[profile] += 1

        output_index += 1

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if output_index % 20 == 0:

            percent = (
                output_index
                / max_frames
                * 100
            )

            print(
                f"Frames: "
                f"{output_index}/"
                f"{max_frames} "
                f"({percent:.1f}%)"
            )

    # =====================================================
    # RELEASE
    # =====================================================

    cap.release()

    # =====================================================
    # WRITE FINAL CHUNKS
    # =====================================================

    for profile in PROFILES:

        if not buffers[profile]:
            continue

        chunk_id = (
            chunk_indices[profile]
        )

        output_path = (
            OUTPUT_DIR
            / profile
            / f"chunk_{chunk_id:04d}.bin"
        )

        (
            raw_size,
            compressed_size
        ) = write_chunk(
            output_path,
            buffers[profile],
            compressors[profile]
        )

        raw_totals[profile] += (
            raw_size
        )

        compressed_totals[profile] += (
            compressed_size
        )

        chunk_indices[profile] += 1

        buffers[profile].clear()

    # =====================================================
    # PROFILE MANIFEST
    # =====================================================

    for profile, size in PROFILES.items():

        width, height = size

        manifest = {
            "profile": profile,
            "width": width,
            "height": height,
            "aspect_ratio": "9:16",
            "fps": FPS,
            "total_frames": output_index,
            "frames_per_chunk": FRAMES_PER_CHUNK,
            "chunks": chunk_indices[profile],
            "format": "RGB565_LE",
            "bytes_per_pixel": 2,
            "compression": "zstd",
            "compression_level": ZSTD_LEVEL,
            "raw_bytes": raw_totals[profile],
            "compressed_bytes": compressed_totals[profile],
            "duration": output_index / FPS,
        }

        manifest_path = (
            OUTPUT_DIR
            / profile
            / "manifest.json"
        )

        with open(
            manifest_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                manifest,
                f,
                indent=2
            )

    # =====================================================
    # MASTER MANIFEST
    # =====================================================

    master = {
        "fps": FPS,
        "duration": min(
            MAX_SECONDS,
            output_index / FPS
        ),
        "total_frames": output_index,
        "aspect_ratio": "9:16",
        "profiles": {}
    }

    for profile, size in PROFILES.items():

        width, height = size

        master["profiles"][profile] = {
            "width": width,
            "height": height,
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

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)

    print(
        f"Output frames: {output_index}"
    )

    print(
        f"Output FPS: {FPS}"
    )

    print(
        f"Duration: "
        f"{output_index / FPS:.2f}s"
    )

    for profile, size in PROFILES.items():

        width, height = size

        compressed_mb = (
            compressed_totals[profile]
            / 1024
            / 1024
        )

        print(
            f"{profile} | "
            f"{width}x{height} | "
            f"{FPS} FPS | "
            f"{chunk_indices[profile]} chunks | "
            f"{compressed_mb:.2f} MB"
        )

    print(
        "Video processing finished."
    )


if __name__ == "__main__":
    main()
