#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture camera images by pressing q."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Directory where captured images will be saved.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    return parser.parse_args()


def choose_output_dir(given_path: str | None) -> Path:
    if given_path:
        path_text = given_path
    else:
        path_text = input("Image save directory: ").strip()

    if not path_text:
        raise ValueError("The image save directory cannot be empty.")

    output_dir = Path(path_text).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    args = parse_args()

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

    print(f"Saving images to: {output_dir}")
    print("Press q to capture an image. Press ESC to exit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break

            preview = frame.copy()
            cv2.putText(
                preview,
                "q: capture | ESC: exit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("CNN Dataset Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_path = output_dir / f"image_{timestamp}.jpg"

                resized_frame = cv2.resize(frame, (640, 480))

                if cv2.imwrite(str(image_path), resized_frame):
                    print(f"Saved: {image_path}")
                else:
                    print(f"Failed to save: {image_path}")

            elif key == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
