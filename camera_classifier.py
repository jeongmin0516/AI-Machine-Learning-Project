#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from kitchen_model import KitchenToolCNN


DEFAULT_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show CNN predictions on a live camera preview."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/kitchen_cnn.pt"),
    )
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--threshold", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--smoothing", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("recordings/classifier"),
        help="Directory for annotated preview recordings.",
    )
    parser.add_argument(
        "--recording-fps",
        type=float,
        default=30.0,
        help="FPS written to recorded videos.",
    )
    return parser.parse_args()


def preprocess(
    frame: np.ndarray,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    if device.type == "cuda":
        tensor = torch.from_numpy(frame).to(device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor[:, [2, 1, 0]]
        tensor = F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return tensor.div_(127.5).sub_(1.0)

    image = cv2.resize(
        frame,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).contiguous()
    return tensor.unsqueeze(0)


def create_label_overlay(
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
) -> tuple[np.ndarray, int]:
    image = Image.new("RGBA", (418, 58), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, 417, 57), radius=10, fill=(0, 0, 0, 180))
    draw.text((12, 8), label, font=font, fill=(*color, 255))
    text_box = draw.textbbox((12, 8), label, font=font)
    confidence_x = min(text_box[2] + 18, 300)
    return np.asarray(image), confidence_x


def draw_prediction(
    frame: np.ndarray,
    overlay: np.ndarray,
    confidence_x: int,
    confidence: float,
) -> np.ndarray:
    preview = frame.copy()
    region = preview[12:70, 12:430]
    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    overlay_bgr = overlay[..., :3][..., ::-1].astype(np.float32)
    region[:] = (
        overlay_bgr * alpha + region.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    cv2.putText(
        preview,
        f"{confidence * 100:.1f}%",
        (12 + confidence_x, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 255, 80),
        2,
        cv2.LINE_AA,
    )
    return preview


def main() -> None:
    args = parse_args()
    if args.smoothing <= 0:
        raise SystemExit("--smoothing must be greater than zero.")
    if args.recording_fps <= 0:
        raise SystemExit("--recording-fps must be greater than zero.")

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    checkpoint = torch.load(
        args.model.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    classes = checkpoint["classes"]
    image_size = int(checkpoint["image_size"])
    model = KitchenToolCNN(len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    history: deque[torch.Tensor] = deque(maxlen=args.smoothing)
    font = ImageFont.truetype(str(args.font), 30)
    label_overlays = [
        create_label_overlay(class_name, (80, 255, 80), font)
        for class_name in classes
    ]

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(f"Could not open camera index {args.camera}.")

    print(f"Classes: {classes}")
    print(f"Device: {device}")
    print(f"Smoothing frames: {args.smoothing}")
    print("Press r to start/stop recording. Press q or ESC to exit.")

    writer: cv2.VideoWriter | None = None
    recording_path: Path | None = None
    recorded_frames = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break

            with torch.inference_mode():
                tensor = preprocess(frame, image_size, device)
                if device.type == "cpu":
                    tensor = tensor.to(device)
                with torch.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                ):
                    logits = model(tensor)
                probabilities = torch.softmax(logits.float(), dim=1)[0]
            history.append(probabilities)
            smoothed = torch.stack(tuple(history)).mean(dim=0)
            confidence_tensor, predicted_tensor = smoothed.max(dim=0)
            predicted_index = int(predicted_tensor.item())
            confidence = float(confidence_tensor.item())

            overlay, confidence_x = label_overlays[predicted_index]
            preview = draw_prediction(
                frame,
                overlay,
                confidence_x,
                confidence,
            )

            if writer is not None:
                cv2.circle(
                    preview,
                    (24, preview.shape[0] - 26),
                    9,
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    preview,
                    f"REC  frames: {recorded_frames + 1}  |  r: stop",
                    (44, preview.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(preview)
                recorded_frames += 1
            else:
                cv2.putText(
                    preview,
                    "r: start recording  |  q/ESC: exit",
                    (18, preview.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Kitchen Tool CNN", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                if writer is None:
                    output_dir = args.recordings_dir.expanduser().resolve()
                    output_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    recording_path = output_dir / f"classifier_{timestamp}.mp4"
                    height, width = preview.shape[:2]
                    writer = cv2.VideoWriter(
                        str(recording_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.recording_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        writer.release()
                        writer = None
                        raise RuntimeError(
                            f"Could not create video file: {recording_path}"
                        )
                    recorded_frames = 0
                    print(f"Recording started: {recording_path}")
                else:
                    writer.release()
                    writer = None
                    print(
                        f"Saved {recorded_frames} frames: {recording_path}"
                    )
                    recording_path = None
                    recorded_frames = 0
            elif key in (ord("q"), 27):
                break
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved {recorded_frames} frames: {recording_path}")
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
