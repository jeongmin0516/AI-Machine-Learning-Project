#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Press r to start recording and q to stop and save."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Directory where the recorded video will be saved.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested video width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested video height (default: 480).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS (default: 30).",
    )
    return parser.parse_args()


def choose_output_dir(given_path: str | None) -> Path:
    path_text = given_path or input("Video save directory: ").strip()
    if not path_text:
        raise ValueError("The video save directory cannot be empty.")

    output_dir = Path(path_text).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("Width, height, and FPS must be positive.")

    try:
        output_dir = choose_output_dir(args.output_dir)
    except (OSError, ValueError) as error:
        print(f"Could not prepare the save directory: {error}")
        raise SystemExit(1) from error

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        print(f"Could not open camera index {args.camera}.")
        raise SystemExit(1)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ok, frame = camera.read()
    if not ok:
        camera.release()
        print("Could not read the first frame from the camera.")
        raise SystemExit(1)

    frame_height, frame_width = frame.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = output_dir / f"video_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        args.fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        camera.release()
        writer.release()
        print(f"Could not create video file: {video_path}")
        raise SystemExit(1)

    print(f"Video will be saved to: {video_path}")
    print("Press r to start recording. Press ESC to exit.")

    frame_count = 0
    is_recording = False
    try:
        while True:
            preview = frame.copy()

            if is_recording:
                writer.write(frame)
                frame_count += 1
                cv2.circle(preview, (25, 25), 9, (0, 0, 255), -1)
                cv2.putText(
                    preview,
                    f"REC  frames: {frame_count}  |  q/ESC: save and exit",
                    (45, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    preview,
                    "READY  |  r: start recording  |  ESC: exit",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("CNN Dataset Video Recorder", preview)

            key = cv2.waitKey(1) & 0xFF
            if not is_recording and key == ord("r"):
                is_recording = True
                print("Recording started.")
            elif is_recording and key == ord("q"):
                break
            elif key == 27:
                break

            ok, frame = camera.read()
            if not ok:
                print("Camera frame reading stopped.")
                break

    finally:
        camera.release()
        writer.release()
        cv2.destroyAllWindows()

    if frame_count > 0:
        print(f"Saved {frame_count} frames: {video_path}")
    else:
        video_path.unlink(missing_ok=True)
        print("Recording was not started. No video was saved.")


if __name__ == "__main__":
    main()
