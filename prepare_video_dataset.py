#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


VIDEO_PATTERN = re.compile(
    r"^(?P<class_name>[^_]+)_(?P<object_id>\d+)_(?P<background>.+)_(?P<take>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leak-resistant image dataset from class videos."
    )
    parser.add_argument("--videos", type=Path, default=Path("raw_videos"))
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--backgrounds",
        nargs="+",
        default=["흰색", "나무색"],
        help="Only include these background names.",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=15,
        help="Save every Nth frame from each video (default: 15).",
    )
    parser.add_argument(
        "--trim-percent",
        type=float,
        default=5.0,
        help="Ignore this percentage at both ends of each video (default: 5).",
    )
    parser.add_argument(
        "--max-images-per-video",
        type=int,
        help="Optional maximum number of saved frames from one video.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the existing output directory before extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.every_n_frames <= 0:
        raise SystemExit("--every-n-frames must be greater than zero.")
    if not 0.0 <= args.trim_percent < 50.0:
        raise SystemExit("--trim-percent must be at least 0 and less than 50.")
    if args.max_images_per_video is not None and args.max_images_per_video <= 0:
        raise SystemExit("--max-images-per-video must be greater than zero.")
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(ratio <= 0.0 for ratio in ratios):
        raise SystemExit("All split ratios must be greater than zero.")
    if not abs(sum(ratios) - 1.0) < 1e-6:
        raise SystemExit("--train-ratio, --val-ratio, and --test-ratio must sum to 1.")

    videos_dir = args.videos.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not videos_dir.is_dir():
        raise SystemExit(f"Video directory does not exist: {videos_dir}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    used_videos: dict[tuple[str, str], int] = defaultdict(int)
    skipped = 0

    for video_path in sorted(videos_dir.rglob("*.mp4")):
        match = VIDEO_PATTERN.match(video_path.stem)
        if match is None:
            print(f"Skipping unrecognized filename: {video_path.name}")
            skipped += 1
            continue

        class_name = match.group("class_name")
        background = match.group("background")
        if background not in args.backgrounds:
            continue

        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        trim_frames = int(frame_count * args.trim_percent / 100.0)
        start_frame = trim_frames
        end_frame = max(start_frame, frame_count - trim_frames - 1)
        usable_frames = end_frame - start_frame + 1
        train_end = start_frame + int(usable_frames * args.train_ratio)
        val_end = train_end + int(usable_frames * args.val_ratio)
        sampled_indices = list(
            range(start_frame, end_frame + 1, args.every_n_frames)
        )
        if (
            args.max_images_per_video is not None
            and len(sampled_indices) > args.max_images_per_video
        ):
            last = len(sampled_indices) - 1
            selected_positions = {
                round(index * last / (args.max_images_per_video - 1))
                for index in range(args.max_images_per_video)
            } if args.max_images_per_video > 1 else {last // 2}
            sampled_indices = [
                sampled_indices[index] for index in sorted(selected_positions)
            ]
        sampled_index_set = set(sampled_indices)

        saved_total = 0
        saved_by_split: dict[str, int] = defaultdict(int)
        frame_index = start_frame
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        try:
            while frame_index <= end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in sampled_index_set:
                    if frame_index < train_end:
                        split = "train"
                    elif frame_index < val_end:
                        split = "val"
                    else:
                        split = "test"
                    class_dir = output_dir / split / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)
                    output_path = class_dir / (
                        f"{video_path.stem}_frame_{frame_index:06d}.jpg"
                    )
                    if cv2.imwrite(
                        str(output_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92],
                    ):
                        saved_total += 1
                        saved_by_split[split] += 1
                    if saved_total >= len(sampled_indices):
                        break
                frame_index += 1
        finally:
            capture.release()

        for split, saved in saved_by_split.items():
            counts[(split, class_name)] += saved
            used_videos[(split, class_name)] += 1

    classes = sorted({class_name for _, class_name in counts})
    print(f"Dataset written to: {output_dir}")
    print(f"Backgrounds: {', '.join(args.backgrounds)}")
    print(f"Frame interval: {args.every_n_frames}")
    print(
        "Split ratios: "
        f"train={args.train_ratio:.2f}, "
        f"val={args.val_ratio:.2f}, "
        f"test={args.test_ratio:.2f}"
    )
    print("split\tclass\tvideos\timages")
    for split in ("train", "val", "test"):
        for class_name in classes:
            print(
                f"{split}\t{class_name}\t"
                f"{used_videos[(split, class_name)]}\t"
                f"{counts[(split, class_name)]}"
            )
    if skipped:
        print(f"Skipped files with unrecognized names: {skipped}")


if __name__ == "__main__":
    main()
