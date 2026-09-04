import cv2
import json
import shutil
from pathlib import Path

import numpy as np
import zstandard as zstd


INPUT_VIDEO = "video.mp4"
OUTPUT_DIR = Path("chunks")

FPS = 8
MAX_SECONDS = 45

# أصغر وأخف بكثير على Roblox
PROFILES = {
    "240p": (426, 240),
    "180p": (320, 180),
}

FRAMES_PER_CHUNK = 4
ZSTD_LEVEL = 3


def frame_to_rgb565(frame_bgr, width, height):
    resized = cv2.resize(
        frame_bgr,
        (width, height),
        interpolation=cv2.INTER_AREA,
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

    return rgb565.astype("<u2", copy=False).tobytes()


def write_chunk(path, frames, compressor):
    raw = b"".join(frames)
    compressed = compressor.compress(raw)

    path.write_bytes(compressed)

    return len(raw), len(compressed)


def main():
    input_path = Path(INPUT_VIDEO)

    if not input_path.exists():
        raise FileNotFoundError(INPUT_VIDEO)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for profile in PROFILES:
        (OUTPUT_DIR / profile).mkdir(
            parents=True,
            exist_ok=True
        )

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
    )

    max_frames = min(
        int(source_duration * FPS),
        FPS * MAX_SECONDS
    )

    compressors = {
        profile: zstd.ZstdCompressor(
            level=ZSTD_LEVEL
        )
        for profile in PROFILES
    }

    buffers = {
        profile: []
        for profile in PROFILES
    }

    indices = {
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

    source_index = 0
    output_index = 0

    sample_step = source_fps / FPS
    next_sample = 0.0

    print("=" * 60)
    print("ROBLOX LOW-LAG VIDEO ENCODER")
    print("=" * 60)
    print("Source FPS:", source_fps)
    print("Output FPS:", FPS)
    print("Max frames:", max_frames)
    print()

    while output_index < max_frames:

        ok, frame = cap.read()

        if not ok:
            break

        current_source = source_index
        source_index += 1

        if current_source + 0.0001 < next_sample:
            continue

        next_sample += sample_step

        for profile, (width, height) in PROFILES.items():

            encoded = frame_to_rgb565(
                frame,
                width,
                height
            )

            buffers[profile].append(encoded)

            if len(buffers[profile]) >= FRAMES_PER_CHUNK:

                chunk_id = indices[profile]

                path = (
                    OUTPUT_DIR /
                    profile /
                    f"chunk_{chunk_id:04d}.bin"
                )

                raw_size, compressed_size = write_chunk(
                    path,
                    buffers[profile],
                    compressors[profile]
                )

                raw_totals[profile] += raw_size
                compressed_totals[profile] += compressed_size

                buffers[profile].clear()
                indices[profile] += 1

        output_index += 1

        if output_index % 20 == 0:

            percent = (
                output_index /
                max_frames *
                100
            )

            print(
                f"{output_index}/{max_frames} "
                f"({percent:.1f}%)"
            )

    cap.release()

    for profile in PROFILES:

        if buffers[profile]:

            chunk_id = indices[profile]

            path = (
                OUTPUT_DIR /
                profile /
                f"chunk_{chunk_id:04d}.bin"
            )

            raw_size, compressed_size = write_chunk(
                path,
                buffers[profile],
                compressors[profile]
            )

            raw_totals[profile] += raw_size
            compressed_totals[profile] += compressed_size

            indices[profile] += 1

    for profile, (width, height) in PROFILES.items():

        manifest = {
            "profile": profile,
            "width": width,
            "height": height,
            "fps": FPS,
            "total_frames": output_index,
            "frames_per_chunk": FRAMES_PER_CHUNK,
            "chunks": indices[profile],
            "format": "RGB565_LE",
            "bytes_per_pixel": 2,
            "compression": "zstd",
            "compression_level": ZSTD_LEVEL,
            "raw_bytes": raw_totals[profile],
            "compressed_bytes": compressed_totals[profile],
        }

        with open(
            OUTPUT_DIR / profile / "manifest.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                manifest,
                f,
                indent=2
            )

    master = {
        "fps": FPS,
        "duration": min(
            MAX_SECONDS,
            output_index / FPS
        ),
        "total_frames": output_index,
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

        raw_mb = raw_totals[profile] / 1024 / 1024
        compressed_mb = (
            compressed_totals[profile] /
            1024 /
            1024
        )

        print(
            f"{profile}: "
            f"{compressed_mb:.1f} MB compressed | "
            f"{indices[profile]} chunks"
        )


if __name__ == "__main__":
    main()
