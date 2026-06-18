#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than 0.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one image for every N frames of a video."
    )
    parser.add_argument("input_video", help="Path to the source video.")
    parser.add_argument("output_dir", help="Directory for extracted images.")
    parser.add_argument(
        "--every-n-frames",
        type=positive_int,
        default=10,
        help="Save one image every N frames (default: 10).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame to inspect (default: 0).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        help="Last frame to inspect, inclusive (default: video end).",
    )
    parser.add_argument(
        "--max-images",
        type=positive_int,
        help="Stop after saving this many images.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100 (default: 95).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_frame < 0:
        raise SystemExit("--start-frame must be 0 or greater.")
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise SystemExit("--end-frame must be greater than or equal to --start-frame.")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100.")

    input_video = Path(args.input_video).expanduser().resolve()
    if not input_video.is_file():
        raise SystemExit(f"Input video does not exist: {input_video}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(input_video))
    if not video.isOpened():
        video.release()
        raise SystemExit(f"Could not open video: {input_video}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    print(f"Source: {input_video}")
    print(f"Frames: {total_frames}, FPS: {fps:.2f}")
    print(
        f"Extracting every {args.every_n_frames} frames "
        f"from frame {args.start_frame}."
    )

    frame_index = args.start_frame
    saved_count = 0

    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if args.end_frame is not None and frame_index > args.end_frame:
                break

            if (frame_index - args.start_frame) % args.every_n_frames == 0:
                image_name = (
                    f"{input_video.stem}_frame_{frame_index:08d}.jpg"
                )
                image_path = output_dir / image_name
                saved = cv2.imwrite(
                    str(image_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                if not saved:
                    raise RuntimeError(f"Failed to save image: {image_path}")

                saved_count += 1
                if args.max_images is not None and saved_count >= args.max_images:
                    break

            frame_index += 1
    finally:
        video.release()

    print(f"Saved {saved_count} images to: {output_dir}")


if __name__ == "__main__":
    main()
