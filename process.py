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

# ------------------------------------------------------------
# Streaming configuration
# ------------------------------------------------------------

# First segment is intentionally short so playback can start
# as quickly as possible.
FIRST_SEGMENT_FRAMES = 8

# Normal target segment length.
# 15 FPS × 15 frames = 1 second.
SEGMENT_FRAMES = 15

# Do not allow a single segment to become huge.
# The encoder will split it earlier if necessary.
MAX_SEGMENT_BYTES = 2_000_000

# ------------------------------------------------------------
# Delta configuration
# ------------------------------------------------------------

# 8x8 tiles are used for spatial delta encoding.
TILE_SIZE = 8

# If a delta frame compresses worse than this percentage
# of the full-frame compressed size, use a full frame instead.
DELTA_MAX_RATIO = 0.90

# Zstandard compression.
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

    return cv2.resize(
        frame,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_AREA
    )


def frame_to_rgb565_array(frame):
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

    return (
        (r5 << 11) |
        (g6 << 5) |
        b5
    ).astype("<u2")


def rgb565_bytes(array):
    return array.tobytes()


def write_u32(file, value):
    file.write(
        struct.pack(
            "<I",
            int(value)
        )
    )


def write_u16(file, value):
    file.write(
        struct.pack(
            "<H",
            int(value)
        )
    )


def build_delta_payload(
    previous,
    current,
    width,
    height
):
    """
    Delta format:

    uint16 changed_tile_count

    For every tile:
        uint8 tile_x
        uint8 tile_y
        raw RGB565 tile bytes

    Tile dimensions are inferred from the image edges.
    """

    tiles_x = math.ceil(
        width / TILE_SIZE
    )

    tiles_y = math.ceil(
        height / TILE_SIZE
    )

    changed_tiles = []

    for tile_y in range(tiles_y):

        y0 = tile_y * TILE_SIZE
        y1 = min(
            y0 + TILE_SIZE,
            height
        )

        for tile_x in range(tiles_x):

            x0 = tile_x * TILE_SIZE
            x1 = min(
                x0 + TILE_SIZE,
                width
            )

            previous_tile = previous[
                y0:y1,
                x0:x1
            ]

            current_tile = current[
                y0:y1,
                x0:x1
            ]

            if np.array_equal(
                previous_tile,
                current_tile
            ):
                continue

            changed_tiles.append(
                (
                    tile_x,
                    tile_y,
                    current_tile.copy()
                )
            )

    payload = bytearray()

    payload.extend(
        struct.pack(
            "<H",
            len(changed_tiles)
        )
    )

    for (
        tile_x,
        tile_y,
        tile
    ) in changed_tiles:

        payload.append(
            tile_x
        )

        payload.append(
            tile_y
        )

        payload.extend(
            tile.astype(
                "<u2"
            ).tobytes()
        )

    return bytes(payload)


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
# SEGMENT WRITER
# ============================================================

def write_segment(
    profile_dir,
    segment_index,
    encoded_frames
):
    """
    Segment format:

    uint32 frame_count

    repeated per frame:
        uint8 frame_type
            0 = full keyframe
            1 = delta frame

        uint32 compressed_size

        compressed_data
    """

    filename = (
        f"segment_{segment_index:04d}.bin"
    )

    path = os.path.join(
        profile_dir,
        filename
    )

    with open(
        path,
        "wb"
    ) as file:

        write_u32(
            file,
            len(encoded_frames)
        )

        for frame_type, data in encoded_frames:

            file.write(
                bytes([frame_type])
            )

            write_u32(
                file,
                len(data)
            )

            file.write(
                data
            )

    size = os.path.getsize(
        path
    )

    return {
        "index": segment_index,
        "filename": filename,
        "frame_count": len(encoded_frames),
        "bytes": size
    }


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
    print("=" * 70)
    print("PROCESSING:", profile_name)
    print(
        "Resolution:",
        f"{width}x{height}"
    )
    print("FPS:", FPS)
    print("First segment:", FIRST_SEGMENT_FRAMES)
    print("Normal segment:", SEGMENT_FRAMES)
    print(
        "Max segment bytes:",
        MAX_SEGMENT_BYTES
    )
    print("=" * 70)

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

    expected_total_frames = max(
        1,
        int(
            math.floor(
                duration * FPS
            )
        )
    )

    print(
        "Source FPS:",
        f"{source_fps:.3f}"
    )

    print(
        "Duration:",
        f"{duration:.2f}s"
    )

    print(
        "Expected output frames:",
        expected_total_frames
    )

    compressor = zstd.ZstdCompressor(
        level=ZSTD_LEVEL
    )

    # --------------------------------------------------------
    # Read output frames
    # --------------------------------------------------------

    output_frames = []

    sample_step = (
        source_fps /
        FPS
    )

    next_sample_source_frame = 0.0

    source_index = 0

    while (
        len(output_frames)
        < expected_total_frames
    ):

        ok, frame = cap.read()

        if not ok:
            break

        current_source_index = (
            source_index
        )

        source_index += 1

        if (
            current_source_index + 1e-9
            < next_sample_source_frame
        ):
            continue

        frame = crop_to_portrait(
            frame,
            width,
            height
        )

        rgb565 = frame_to_rgb565_array(
            frame
        )

        output_frames.append(
            rgb565
        )

        next_sample_source_frame += (
            sample_step
        )

        if (
            len(output_frames) % 30 == 0
            or len(output_frames)
            == expected_total_frames
        ):

            print(
                f"Frames: "
                f"{len(output_frames)}/"
                f"{expected_total_frames}"
            )

    cap.release()

    if not output_frames:
        raise RuntimeError(
            "No output frames generated."
        )

    total_frames = len(
        output_frames
    )

    # --------------------------------------------------------
    # Encode into segments
    # --------------------------------------------------------

    segments = []

    segment_frames = []

    segment_start_frame = 0

    segment_target_frames = (
        FIRST_SEGMENT_FRAMES
        if len(segments) == 0
        else SEGMENT_FRAMES
    )

    segment_index = 0

    global_frame_index = 0

    while global_frame_index < total_frames:

        # Start every segment with a full keyframe.
        keyframe = output_frames[
            global_frame_index
        ]

        full_raw = rgb565_bytes(
            keyframe
        )

        full_compressed = (
            compressor.compress(
                full_raw
            )
        )

        segment_frames = [
            (
                0,
                full_compressed
            )
        ]

        segment_bytes = (
            1 +
            4 +
            len(full_compressed)
        )

        previous_frame = keyframe

        global_frame_index += 1

        frame_count = 1

        # ----------------------------------------------------
        # Add delta/full frames
        # ----------------------------------------------------

        while (
            global_frame_index
            < total_frames

            and frame_count
            < segment_target_frames
        ):

            current_frame = (
                output_frames[
                    global_frame_index
                ]
            )

            delta_payload = (
                build_delta_payload(
                    previous_frame,
                    current_frame,
                    width,
                    height
                )
            )

            delta_compressed = (
                compressor.compress(
                    delta_payload
                )
            )

            current_full_compressed = (
                compressor.compress(
                    rgb565_bytes(
                        current_frame
                    )
                )
            )

            # Use delta only when it is meaningfully smaller.
            if (
                len(delta_compressed)
                <=
                len(current_full_compressed)
                * DELTA_MAX_RATIO
            ):

                frame_type = 1
                frame_data = (
                    delta_compressed
                )

            else:

                frame_type = 0
                frame_data = (
                    current_full_compressed
                )

            projected_size = (
                segment_bytes
                + 1
                + 4
                + len(frame_data)
            )

            if (
                projected_size
                > MAX_SEGMENT_BYTES
            ):
                break

            segment_frames.append(
                (
                    frame_type,
                    frame_data
                )
            )

            segment_bytes = (
                projected_size
            )

            previous_frame = (
                current_frame
            )

            global_frame_index += 1
            frame_count += 1

        segment_info = write_segment(
            profile_dir,
            segment_index,
            segment_frames
        )

        segment_info[
            "start_frame"
        ] = segment_start_frame

        segment_info[
            "end_frame"
        ] = (
            segment_start_frame
            +
            frame_count
            - 1
        )

        segment_info[
            "duration"
        ] = (
            frame_count /
            FPS
        )

        full_count = sum(
            1
            for frame_type, _
            in segment_frames
            if frame_type == 0
        )

        delta_count = sum(
            1
            for frame_type, _
            in segment_frames
            if frame_type == 1
        )

        segment_info[
            "keyframes"
        ] = full_count

        segment_info[
            "delta_frames"
        ] = delta_count

        segments.append(
            segment_info
        )

        print(
            f"Segment {segment_index:04d}: "
            f"{frame_count} frames, "
            f"{segment_info['bytes'] / 1024 / 1024:.2f} MB, "
            f"{full_count} full, "
            f"{delta_count} delta"
        )

        segment_index += 1

        segment_start_frame += (
            frame_count
        )

        # After the first segment, normal target size.
        segment_target_frames = (
            SEGMENT_FRAMES
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    raw_bytes_per_frame = (
        width *
        height *
        2
    )

    raw_bytes = (
        raw_bytes_per_frame *
        total_frames
    )

    total_segment_bytes = sum(
        segment["bytes"]
        for segment in segments
    )

    manifest = {
        "version": 2,
        "profile": profile_name,

        "width": width,
        "height": height,

        "aspect_ratio": "9:16",

        "fps": FPS,
        "total_frames": total_frames,

        "duration": (
            total_frames /
            FPS
        ),

        "format": "RGB565_LE",
        "bytes_per_pixel": 2,

        "compression": "zstd",
        "compression_level": ZSTD_LEVEL,

        "tile_size": TILE_SIZE,

        "segment_count": len(
            segments
        ),

        "first_segment_frames":
            FIRST_SEGMENT_FRAMES,

        "target_segment_frames":
            SEGMENT_FRAMES,

        "max_segment_bytes":
            MAX_SEGMENT_BYTES,

        "raw_bytes": raw_bytes,

        "segment_bytes":
            total_segment_bytes,

        "segments": segments
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

    print()
    print("=" * 70)
    print("PROFILE COMPLETE")
    print("=" * 70)

    print(
        "Frames:",
        total_frames
    )

    print(
        "Segments:",
        len(segments)
    )

    print(
        "Duration:",
        f'{manifest["duration"]:.2f}s'
    )

    print(
        "Raw size:",
        f"{raw_bytes / 1024 / 1024:.2f} MB"
    )

    print(
        "Segment size:",
        f"{total_segment_bytes / 1024 / 1024:.2f} MB"
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

    manifests = {}

    for profile_name, (
        width,
        height
    ) in PROFILES.items():

        manifests[
            profile_name
        ] = process_profile(
            video_path,
            profile_name,
            width,
            height
        )

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
            manifests,
            file,
            indent=2
        )

    print()
    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

