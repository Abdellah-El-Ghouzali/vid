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

# فيديو Shorts
FPS = 8
MAX_SECONDS = 45

# Portrait 9:16
PROFILES = {
    "240p": (240, 426),
}

FRAMES_PER_CHUNK = 4
ZSTD_LEVEL = 3


# =========================================================
# CROP + RESIZE + RGB565
# =========================================================

def frame_to_rgb565(frame_bgr, width, height):
    source_height, source_width = frame_bgr.shape[:2]

    target_ratio = width / height
    source_ratio = source_width / source_height

    # -----------------------------------------------------
    # Crop to target ratio
    # -----------------------------------------------------

    if source_ratio > target_ratio:
        # الفيديو أعرض من 9:16
        new_width = int(source_height * target_ratio)

        left = (source_width - new_width) // 2

        frame_bgr = frame_bgr[
            :,
            left:left + new_width
        ]

    elif source_ratio < target_ratio:
        # الفيديو أطول من 9:16
        new_height = int(source_width / target_ratio)

        top = (source_height - new_height) // 2

        frame_bgr = frame_bgr[
            top:top + new_height,
            :
        ]

    # -----------------------------------------------------
    # Resize
    # -----------------------------------------------------

    resized = cv2.resize(
        frame_bgr,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    # -----------------------------------------------------
    # BGR -> RGB565
    # -----------------------------------------------------

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

    return rgb565.astype(
        "<u2",
        copy=False
    ).tobytes()


# =========================================================
# CHUNK
# =========================================================

def write_chunk(path, frames, compressor):
    raw = b"".join(frames)

    compressed = compressor.compress(raw)

    path.write_bytes(compressed)

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

    # -----------------------------------------------------
    # Remove previous generated output
    # -----------------------------------------------------

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Create profile directories
    # -----------------------------------------------------

    for profile in PROFILES:
        (
            OUTPUT_DIR / profile
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # -----------------------------------------------------
    # Open video
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not source_fps or source_fps <= 0:
        source_fps = FPS

    source_frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    source_duration = (
        source_frame_count /
        source_fps
    )

    max_frames = min(
        int(source_duration * FPS),
        FPS * MAX_SECONDS
    )

    # -----------------------------------------------------
    # Compressors
    # -----------------------------------------------------

    compressors = {
        profile: zstd.ZstdCompressor(
            level=ZSTD_LEVEL
        )
        for profile in PROFILES
    }

    # -----------------------------------------------------
    # Buffers
    # -----------------------------------------------------

    buffers = {
        profile: []
        for profile in PROFILES
    }

    chunk_indices = {
        profile: 0
        for profile in PROFILES
    }

    raw_totals = {
        profile: 0
        for profile in PROFILES
    }

    compressed_totals = {
        profile: 0
        for profile in PROFILES
    }

    # -----------------------------------------------------
    # Sampling
    # -----------------------------------------------------

    source_index = 0
    output_index = 0

    sample_step =
        source_fps / FPS

    next_sample = 0.0

    print("=" * 60)
    print("ROBLOX PORTRAIT VIDEO ENCODER")
    print("=" * 60)
    print(f"Source FPS : {source_fps:.2f}")
    print(f"Output FPS : {FPS}")
    print(f"Max frames : {max_frames}")
    print()

    # -----------------------------------------------------
    # Processing
    # -----------------------------------------------------

    while output_index < max_frames:

        ok, frame = cap.read()

        if not ok:
            break

        current_source =
            source_index

        source_index += 1

        # Sampling
        if (
            current_source + 0.0001
            < next_sample
        ):
            continue

        next_sample += sample_step

        # -------------------------------------------------
        # Encode every profile
        # -------------------------------------------------

        for profile, (width, height) in PROFILES.items():

            encoded =
                frame_to_rgb565(
                    frame,
                    width,
                    height
                )

            buffers[profile].append(
                encoded
            )

            # -------------------------------------------------
            # Write chunk
            # -------------------------------------------------

            if (
                len(buffers[profile])
                >= FRAMES_PER_CHUNK
            ):

                chunk_id =
                    chunk_indices[profile]

                output_path = (
                    OUTPUT_DIR /
                    profile /
                    f"chunk_{chunk_id:04d}.bin"
                )

                raw_size, compressed_size = (
                    write_chunk(
                        output_path,
                        buffers[profile],
                        compressors[profile]
                    )
                )

                raw_totals[profile] += raw_size
                compressed_totals[profile] += compressed_size

                buffers[profile].clear()

                chunk_indices[profile] += 1

        output_index += 1

        if output_index % 20 == 0:

            percent = (
                output_index /
                max_frames *
                100
            )

            print(
                f"Frames: "
                f"{output_index}/"
                f"{max_frames} "
                f"({percent:.1f}%)"
            )

    cap.release()

    # =====================================================
    # FINAL CHUNKS
    # =====================================================

    for profile in PROFILES:

        if buffers[profile]:

            chunk_id =
                chunk_indices[profile]

            output_path = (
                OUTPUT_DIR /
                profile /
                f"chunk_{chunk_id:04d}.bin"
            )

            raw_size, compressed_size = (
                write_chunk(
                    output_path,
                    buffers[profile],
                    compressors[profile]
                )
            )

            raw_totals[profile] += raw_size
            compressed_totals[profile] += compressed_size

            chunk_indices[profile] += 1

    # =====================================================
    # PROFILE MANIFESTS
    # =====================================================

    for profile, (width, height) in PROFILES.items():

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
        }

        manifest_path = (
            OUTPUT_DIR /
            profile /
            "manifest.json"
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

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)

    for profile in PROFILES:

        compressed_mb = (
            compressed_totals[profile] /
            1024 /
            1024
        )

        print(
            f"{profile} | "
            f"Resolution: "
            f"{PROFILES[profile][0]}x"
            f"{PROFILES[profile][1]} | "
            f"FPS: {FPS} | "
            f"Chunks: "
            f"{chunk_indices[profile]} | "
            f"Compressed: "
            f"{compressed_mb:.1f} MB"
        )


if __name__ == "__main__":
    main()
