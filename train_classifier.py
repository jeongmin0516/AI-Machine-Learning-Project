#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kitchen_model import KitchenToolCNN


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the kitchen-tool CNN.")
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("models/kitchen_cnn.pt"))
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="OpenCV CPU threads per process.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--rotation-degrees",
        type=float,
        default=180.0,
        help="Maximum absolute training rotation in degrees.",
    )
    parser.add_argument(
        "--horizontal-flip-probability",
        type=float,
        default=0.0,
        help="Training-time horizontal flip probability.",
    )
    parser.add_argument(
        "--grayscale-probability",
        type=float,
        default=0.15,
        help="Training-time grayscale conversion probability.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_worker(_: int, opencv_threads: int) -> None:
    cv2.setNumThreads(opencv_threads)
    torch.set_num_threads(1)


def normalize(image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    array = rgb.astype(np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()


def augment_geometry(
    image: np.ndarray,
    rotation_degrees: float,
    horizontal_flip_probability: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    if random.random() < 0.9:
        angle = random.uniform(-rotation_degrees, rotation_degrees)
        scale = random.uniform(0.88, 1.05)
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += random.uniform(-0.04, 0.04) * width
        matrix[1, 2] += random.uniform(-0.04, 0.04) * height
        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    if random.random() < 0.15:
        jitter = 0.02
        corners = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        offsets = np.column_stack(
            (
                np.random.uniform(-jitter, jitter, 4) * width,
                np.random.uniform(-jitter, jitter, 4) * height,
            )
        ).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(corners, corners + offsets)
        image = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    if random.random() < horizontal_flip_probability:
        image = cv2.flip(image, 1)
    return image


def augment_appearance(
    image: np.ndarray,
    grayscale_probability: float,
) -> np.ndarray:
    if random.random() < 0.8:
        alpha = random.uniform(0.82, 1.18)
        beta = random.uniform(-18.0, 18.0)
        image = np.clip(
            image.astype(np.float32) * alpha + beta,
            0,
            255,
        ).astype(np.uint8)

    if random.random() < 0.3:
        gamma = random.uniform(0.8, 1.2)
        lookup = np.clip(
            (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0,
            0,
            255,
        ).astype(np.uint8)
        image = cv2.LUT(image, lookup)

    if random.random() < 0.35:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-5.0, 5.0)) % 180
        hsv[..., 1] *= random.uniform(0.75, 1.25)
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < 0.2:
        channel_gains = np.random.uniform(0.9, 1.1, 3).astype(np.float32)
        image = np.clip(
            image.astype(np.float32) * channel_gains,
            0,
            255,
        ).astype(np.uint8)

    if random.random() < 0.12:
        height, width = image.shape[:2]
        horizontal = random.random() < 0.5
        ramp = np.linspace(
            random.uniform(0.65, 0.9),
            1.0,
            width if horizontal else height,
            dtype=np.float32,
        )
        if random.random() < 0.5:
            ramp = ramp[::-1]
        illumination = ramp[None, :, None] if horizontal else ramp[:, None, None]
        image = np.clip(
            image.astype(np.float32) * illumination,
            0,
            255,
        ).astype(np.uint8)

    degradation = random.random()
    if degradation < 0.1:
        image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.2, 1.0))
    elif degradation < 0.25:
        noise = np.random.normal(
            0.0,
            random.uniform(2.0, 7.0),
            image.shape,
        ).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.1:
        height, width = image.shape[:2]
        erase_height = random.randint(max(1, height // 25), max(2, height // 10))
        erase_width = random.randint(max(1, width // 25), max(2, width // 10))
        top = random.randint(0, height - erase_height)
        left = random.randint(0, width - erase_width)
        fill = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)
        image[top : top + erase_height, left : left + erase_width] = fill

    if random.random() < grayscale_probability:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return image


class ToolDataset(Dataset):
    def __init__(
        self,
        root: Path,
        classes: list[str],
        image_size: int,
        training: bool,
        rotation_degrees: float = 180.0,
        horizontal_flip_probability: float = 0.0,
        grayscale_probability: float = 0.15,
    ) -> None:
        self.image_size = image_size
        self.training = training
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip_probability = horizontal_flip_probability
        self.grayscale_probability = grayscale_probability
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.samples: list[tuple[Path, int]] = []
        for class_name in classes:
            class_dir = root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((path, self.class_to_index[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        if self.training:
            image = augment_geometry(
                image,
                self.rotation_degrees,
                self.horizontal_flip_probability,
            )
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        if self.training:
            image = augment_appearance(image, self.grayscale_probability)
        return normalize(image), label


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = device.type == "cuda" and scaler is not None
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if training and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            elif training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    return total_loss / sample_count, correct / sample_count


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    for images, labels in loader:
        probabilities = torch.softmax(
            model(images.to(device, non_blocking=True)),
            dim=1,
        )
        confidence, predicted = probabilities.max(dim=1)
        targets.extend(labels.tolist())
        predictions.extend(predicted.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
    return targets, predictions, confidences


def main() -> None:
    args = parse_args()
    if args.rotation_degrees < 0:
        raise SystemExit("--rotation-degrees must be non-negative.")
    if not 0.0 <= args.horizontal_flip_probability <= 1.0:
        raise SystemExit("--horizontal-flip-probability must be between 0 and 1.")
    if not 0.0 <= args.grayscale_probability <= 1.0:
        raise SystemExit("--grayscale-probability must be between 0 and 1.")
    if args.workers < 0:
        raise SystemExit("--workers must be non-negative.")
    if args.opencv_threads < 1:
        raise SystemExit("--opencv-threads must be at least 1.")
    set_seed(args.seed)
    cv2.setNumThreads(args.opencv_threads)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    data_dir = args.data.expanduser().resolve()
    classes = sorted(path.name for path in (data_dir / "train").iterdir() if path.is_dir())
    if len(classes) < 2:
        raise SystemExit(f"Need at least two class directories in {data_dir / 'train'}")

    datasets = {
        split: ToolDataset(
            data_dir / split,
            classes,
            args.image_size,
            training=(split == "train"),
            rotation_degrees=args.rotation_degrees,
            horizontal_flip_probability=args.horizontal_flip_probability,
            grayscale_probability=args.grayscale_probability,
        )
        for split in ("train", "val", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise SystemExit("Train, validation, and test datasets must all contain images.")

    pin_memory = device.type == "cuda"
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.workers,
            pin_memory=pin_memory,
            persistent_workers=args.workers > 0,
            worker_init_fn=(
                partial(configure_worker, opencv_threads=args.opencv_threads)
                if args.workers > 0
                else None
            ),
        )
        for split, dataset in datasets.items()
    }

    train_counts = Counter(label for _, label in datasets["train"].samples)
    class_weights = torch.tensor(
        [len(datasets["train"]) / (len(classes) * train_counts[index]) for index in range(len(classes))],
        dtype=torch.float32,
        device=device,
    )
    model = KitchenToolCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03) # 0.1이었음
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.4,
        patience=2,
        min_lr=1e-6,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and not args.no_amp
        else None
    )

    print(f"Device: {device}, AMP: {scaler is not None}")
    print(f"Classes: {classes}")
    print("Samples:", {name: len(dataset) for name, dataset in datasets.items()})
    print(
        "Augmentation: "
        f"rotation=+/-{args.rotation_degrees:g} degrees, "
        f"horizontal_flip_probability={args.horizontal_flip_probability:g}, "
        f"grayscale_probability={args.grayscale_probability:g}"
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = 0.0
    stale_epochs = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, device, optimizer, scaler
        )
        val_loss, val_acc = run_epoch(
            model, loaders["val"], criterion, device
        )
        scheduler.step(val_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:02d}/{args.epochs}  "
            f"train loss={train_loss:.4f} acc={train_acc:.3f}  "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}  "
            f"lr={learning_rate:.2e}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    model.load_state_dict(best_state)
    targets, predictions, confidences = predict(model, loaders["test"], device)
    test_accuracy = float(np.mean(np.equal(targets, predictions)))
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(classes))))
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_name": "KitchenToolCNN",
        "model_state": model.cpu().state_dict(),
        "classes": classes,
        "image_size": args.image_size,
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "recommended_threshold": 0.75,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_accuracy,
        "augmentation": {
            "rotation_degrees": args.rotation_degrees,
            "horizontal_flip_probability": args.horizontal_flip_probability,
            "grayscale_probability": args.grayscale_probability,
        },
    }
    torch.save(checkpoint, args.output)

    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "classes": classes,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_acc,
                "test_accuracy": test_accuracy,
                "mean_test_confidence": float(np.mean(confidences)),
                "augmentation": checkpoint["augmentation"],
                "confusion_matrix": matrix.tolist(),
                "elapsed_seconds": time.time() - started,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print("Confusion matrix:")
    print(matrix)
    print(report)
    print(f"Saved model: {args.output.resolve()}")
    print(f"Saved metrics: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
