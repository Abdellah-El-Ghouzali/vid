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

# Number of large pack files.
# 5 = maximum 5 HTTP downloads for the video.
PACK_COUNT = 5

# Zstandard compression level.
ZSTD_LEVEL = 3

OUTPUT_DIR = "chunks"


# ============================================================
# HELPERS
# ============================================================

def crop_to_portrait(
    frame,
    target_width,
    target_height
):
    source_height, source_width = frame.shape[:2]

    target_ratio = (
        target_width /
        target_height
    )

    source_ratio = (
        source_width /
        source_height
    )

    if source_ratio > target_ratio:

        new_width = int(
            source_height *
            target_ratio
        )

        left = max(
            (source_width - new_width) // 2,
            0
        )

        frame = frame[
            :,
            left:left + new_width
        ]

    else:

        new_height = int(
            source_width /
            target_ratio
        )

        top = max(
            (source_height - new_height) // 2,
            0
        )

        frame = frame[
            top:top + new_height,
            :
        ]

    frame = cv2.resize(
        frame,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_AREA
    )

    return frame


def frame_to_rgb565(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    r = rgb[:, :, 0].astype(
        np.uint16
    )

    g = rgb[:, :, 1].astype(
        np.uint16
    )

    b = rgb[:, :, 2].astype(
        np.uint16
    )

    r5 = (
        r * 31 + 127
    ) // 255

    g6 = (
        g * 63 + 127
    ) // 255

    b5 = (
        b * 31 + 127
    ) // 255

    packed = (
        (r5 << 11) |
        (g6 << 5) |
        b5
    ).astype("<u2")

    return packed.tobytes()


def write_u32(
    file,
    value
):
    file.write(
        struct.pack(
            "<I",
            value
        )
    )


def clean_output():

    if os.path.exists(
        OUTPUT_DIR
    ):
        shutil.rmtree(
            OUTPUT_DIR
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


# ============================================================
# PROCESS PROFILE
# ============================================================

def process_profile(
    video_path,
    profile_name,
    width,
    height
):

    profile_dir = os.path.join(
        OUTPUT_DIR,
        profile_name
    )

    os.makedirs(
        profile_dir,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print(
        f"Processing: {profile_name}"
    )
    print(
        f"Resolution: {width}x{height}"
    )
    print(
        f"FPS: {FPS}"
    )
    print(
        f"Packs: {PACK_COUNT}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video.mp4"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if (
        not source_fps
        or source_fps <= 0
    ):

        source_fps = 30.0

    source_frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    source_duration = (
        source_frame_count /
        source_fps
        if source_frame_count > 0
        else 0
    )

    duration = min(
        source_duration,
        MAX_SECONDS
    )

    total_frames = max(
        1,
        int(
            math.floor(
                duration * FPS
            )
        )
    )

    print(
        f"Source FPS: {source_fps:.3f}"
    )

    print(
        f"Source duration: "
        f"{source_duration:.2f}s"
    )

    print(
        f"Output duration: "
        f"{duration:.2f}s"
    )

    print(
        f"Total frames: "
        f"{total_frames}"
    )

    # --------------------------------------------------------
    # COMPRESS FRAMES
    # --------------------------------------------------------

    compressor = zstd.ZstdCompressor(
        level=ZSTD_LEVEL
    )

    compressed_frames = []

    sample_step = (
        source_fps /
        FPS
    )

    next_sample_source_frame = 0.0

    source_index = 0
    output_index = 0

    while (
        output_index <
        total_frames
    ):

        ok, frame = cap.read()

        if not ok:
            break

        current_source_index = (
            source_index
        )

        source_index += 1

        if (
            current_source_index
            + 1e-9
            <
            next_sample_source_frame
        ):
            continue

        frame = crop_to_portrait(
            frame,
            width,
            height
        )

        raw_rgb565 = frame_to_rgb565(
            frame
        )

        compressed = (
            compressor.compress(
                raw_rgb565
            )
        )

        compressed_frames.append(
            compressed
        )

        output_index += 1

        next_sample_source_frame += (
            sample_step
        )

        if (
            output_index % 30 == 0
            or output_index == total_frames
        ):

            percent = (
                output_index /
                total_frames *
                100
            )

            print(
                f"Frames: "
                f"{output_index}/"
                f"{total_frames} "
                f"({percent:.1f}%)"
            )

    cap.release()

    if not compressed_frames:

        raise RuntimeError(
            "No frames were generated."
        )

    total_frames = len(
        compressed_frames
    )

    # --------------------------------------------------------
    # PACK DISTRIBUTION
    # --------------------------------------------------------

    actual_pack_count = min(
        PACK_COUNT,
        total_frames
    )

    chunks_per_pack = math.ceil(
        total_frames /
        actual_pack_count
    )

    print()
    print(
        f"Actual pack count: "
        f"{actual_pack_count}"
    )

    print(
        f"Frames per pack: "
        f"{chunks_per_pack}"
    )

    # --------------------------------------------------------
    # WRITE PACKS
    #
    # PACK FORMAT:
    #
    # uint32 frame_count
    #
    # repeated:
    #   uint32 compressed_size
    #   compressed_frame
    #
    # --------------------------------------------------------

    pack_files = []

    for pack_index in range(
        actual_pack_count
    ):

        start_frame = (
            pack_index *
            chunks_per_pack
        )

        end_frame = min(
            start_frame +
            chunks_per_pack,
            total_frames
        )

        frames = compressed_frames[
            start_frame:end_frame
        ]

        filename = (
            f"pack_{pack_index:03d}.bin"
        )

        path = os.path.join(
            profile_dir,
            filename
        )

        print()
        print(
            f"Writing {filename}"
        )

        with open(
            path,
            "wb"
        ) as file:

            # Number of frames
            write_u32(
                file,
                len(frames)
            )

            for frame_data in frames:

                # Compressed frame size
                write_u32(
                    file,
                    len(frame_data)
                )

                # Frame bytes
                file.write(
                    frame_data
                )

        size = os.path.getsize(
            path
        )

        pack_files.append({
            "index": pack_index,
            "filename": filename,
            "start_frame": start_frame,
            "frame_count": len(frames),
            "end_frame": end_frame - 1,
            "bytes": size
        })

        print(
            "Size:",
            f"{size / 1024 / 1024:.2f} MB"
        )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    compressed_bytes = sum(
        len(frame)
        for frame in compressed_frames
    )

    raw_bytes_per_frame = (
        width *
        height *
        2
    )

    raw_bytes = (
        raw_bytes_per_frame *
        total_frames
    )

    manifest = {
        "profile": profile_name,
        "width": width,
        "height": height,
        "aspect_ratio": "9:16",
        "fps": FPS,
        "total_frames": total_frames,

        # One compressed frame per entry.
        "frames_per_chunk": 1,

        # Kept for compatibility.
        "chunks": total_frames,

        # New packed format.
        "packs": actual_pack_count,
        "frames_per_pack": chunks_per_pack,

        "format": "RGB565_LE",
        "bytes_per_pixel": 2,
        "compression": "zstd",
        "compression_level": ZSTD_LEVEL,

        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,

        "duration": (
            total_frames /
            FPS
        ),

        "pack_files": pack_files
    }

    manifest_path = os.path.join(
        profile_dir,
        "manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_pack_size = sum(
        pack["bytes"]
        for pack in pack_files
    )

    print()
    print("=" * 60)
    print(
        f"{profile_name} COMPLETE"
    )
    print("=" * 60)

    print(
        "Resolution:",
        f"{width}x{height}"
    )

    print(
        "FPS:",
        FPS
    )

    print(
        "Frames:",
        total_frames
    )

    print(
        "Packs:",
        actual_pack_count
    )

    print(
        "Duration:",
        f'{manifest["duration"]:.2f}s'
    )

    print(
        "Total packed size:",
        f"{total_pack_size / 1024 / 1024:.2f} MB"
    )

    print(
        "Manifest:",
        manifest_path
    )

    return manifest


# ============================================================
# MAIN
# ============================================================

def main():

    video_path = "video.mp4"

    if not os.path.isfile(
        video_path
    ):

        raise FileNotFoundError(
            "video.mp4 was not found."
        )

    clean_output()

    all_manifests = {}

    for profile_name, (
        width,
        height
    ) in PROFILES.items():

        manifest = process_profile(
            video_path,
            profile_name,
            width,
            height
        )

        all_manifests[
            profile_name
        ] = manifest

    # --------------------------------------------------------
    # MASTER MANIFEST
    # --------------------------------------------------------

    master_manifest_path = os.path.join(
        OUTPUT_DIR,
        "manifest.json"
    )

    with open(
        master_manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_manifests,
            file,
            indent=2
        )

    print()
    print("=" * 60)
    print("ALL PROCESSING COMPLETE")
    print("=" * 60)

    for profile_name, manifest in (
        all_manifests.items()
    ):

        print()
        print(
            f"Profile: {profile_name}"
        )

        print(
            "Resolution:",
            manifest["width"],
            "x",
            manifest["height"]
        )

        print(
            "FPS:",
            manifest["fps"]
        )

        print(
            "Frames:",
            manifest["total_frames"]
        )

        print(
            "Packs:",
            manifest["packs"]
        )

        print(
            "Duration:",
            f'{manifest["duration"]:.2f}s'
        )


if __name__ == "__main__":
    main()

