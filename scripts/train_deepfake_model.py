from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.deepfake_detection import extract_embedding  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Train a deepfake detector with a real vision backbone.")
    parser.add_argument(
        "--dataset-dir",
        default=str(ROOT / "external" / "kaggle" / "deepfake-and-real-images" / "Dataset"),
        help="Dataset root. Supports Kaggle Train/Validation/Test folders or a flat real/fake folder layout.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "deployment" / "deepfake_model"),
        help="Where to save the deepfake model bundle.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=160,
        help="Square resize used for the vision backbone.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding extraction batch size. Keep this modest for CPU laptops.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Zero keeps CPU pressure lower on laptops.",
    )
    parser.add_argument(
        "--backbone-name",
        default="mobilenet_v3_small",
        help="Vision backbone used for feature extraction.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=8,
        help="Minimum number of head-training epochs before target-based early stopping is allowed.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=24,
        help="Hard cap for head-training epochs.",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.9,
        help="Validation accuracy target used for early stopping.",
    )
    parser.add_argument(
        "--target-confidence",
        type=float,
        default=0.9,
        help="Validation average confidence target used for early stopping.",
    )
    parser.add_argument(
        "--max-train-per-label",
        type=int,
        default=0,
        help="Optional cap per label for the training split only. Use 0 to keep all currently loaded Kaggle images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser


def compute_metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
        pos_label=1,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "avg_confidence": float(np.max(probabilities, axis=1).mean()) if len(probabilities) else 0.0,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0,
            target_names=["real", "fake"],
        ),
    }


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], image_size: int, backbone_name: str) -> None:
        self.samples = samples
        self.image_size = image_size
        self.backbone_name = backbone_name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        embedding = extract_embedding(
            image,
            backbone_name=self.backbone_name,
            image_size=self.image_size,
        )
        return embedding, label


def collate_embeddings(batch: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.stack([embedding for embedding, _ in batch]).astype(np.float32)
    labels = np.array([label for _, label in batch], dtype=np.int32)
    return embeddings, labels


def collect_image_paths(split_dir: Path, max_per_label: int, seed: int) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    samples: list[tuple[Path, int]] = []
    for folder_name, label in (("Real", 0), ("Fake", 1), ("real", 0), ("fake", 1)):
        label_dir = split_dir / folder_name
        if not label_dir.exists():
            continue
        image_paths = [
            path for path in sorted(label_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if max_per_label > 0 and len(image_paths) > max_per_label:
            image_paths = rng.sample(image_paths, max_per_label)
        samples.extend((path, label) for path in image_paths)
    if not samples:
        raise FileNotFoundError(f"No labelled images found under split: {split_dir}")
    rng.shuffle(samples)
    return samples


def resolve_split_dirs(dataset_dir: Path) -> tuple[Path, Path, Path]:
    train_dir = dataset_dir / "Train"
    val_dir = dataset_dir / "Validation"
    test_dir = dataset_dir / "Test"
    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        return train_dir, val_dir, test_dir
    raise FileNotFoundError(
        "Expected Kaggle split folders Train/Validation/Test under "
        f"{dataset_dir}, but they were not found."
    )


def extract_split_embeddings(
    samples: list[tuple[Path, int]],
    *,
    image_size: int,
    backbone_name: str,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = ImagePathDataset(samples=samples, image_size=image_size, backbone_name=backbone_name)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_embeddings,
    )
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch_embeddings, batch_labels in loader:
        embeddings.append(batch_embeddings)
        labels.append(batch_labels)
    return np.vstack(embeddings), np.concatenate(labels)


def evaluate_classifier(classifier: SGDClassifier, features: np.ndarray, labels: np.ndarray) -> dict:
    probabilities = classifier.predict_proba(features)
    predictions = np.argmax(probabilities, axis=1)
    return compute_metrics(labels, predictions, probabilities)


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2 or 1)))

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir, val_dir, test_dir = resolve_split_dirs(dataset_dir)
    train_samples = collect_image_paths(train_dir, max_per_label=args.max_train_per_label, seed=args.seed)
    val_samples = collect_image_paths(val_dir, max_per_label=0, seed=args.seed)
    test_samples = collect_image_paths(test_dir, max_per_label=0, seed=args.seed)

    x_train, y_train = extract_split_embeddings(
        train_samples,
        image_size=args.image_size,
        backbone_name=args.backbone_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    x_val, y_val = extract_split_embeddings(
        val_samples,
        image_size=args.image_size,
        backbone_name=args.backbone_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    x_test, y_test = extract_split_embeddings(
        test_samples,
        image_size=args.image_size,
        backbone_name=args.backbone_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0

    train_scaled = (x_train - mean) / scale
    val_scaled = (x_val - mean) / scale
    test_scaled = (x_test - mean) / scale

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        learning_rate="optimal",
        class_weight="balanced",
        random_state=args.seed,
    )

    best_validation_f1 = -1.0
    best_epoch = 0
    best_coefficients: np.ndarray | None = None
    best_intercept: np.ndarray | None = None
    history: list[dict] = []

    indices = np.arange(len(train_scaled))
    classes = np.array([0, 1], dtype=np.int32)
    for epoch in range(args.max_epochs):
        np.random.shuffle(indices)
        epoch_features = train_scaled[indices]
        epoch_labels = y_train[indices]
        if epoch == 0:
            classifier.partial_fit(epoch_features, epoch_labels, classes=classes)
        else:
            classifier.partial_fit(epoch_features, epoch_labels)

        train_metrics = evaluate_classifier(classifier, train_scaled, y_train)
        validation_metrics = evaluate_classifier(classifier, val_scaled, y_val)
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )

        if validation_metrics["f1_fake"] > best_validation_f1:
            best_validation_f1 = float(validation_metrics["f1_fake"])
            best_epoch = epoch + 1
            best_coefficients = classifier.coef_.copy()
            best_intercept = classifier.intercept_.copy()

        meets_targets = (
            float(validation_metrics["accuracy"]) >= args.target_accuracy
            and float(validation_metrics["avg_confidence"]) >= args.target_confidence
        )
        if (epoch + 1) >= args.num_epochs and meets_targets:
            break

    if best_coefficients is None or best_intercept is None:
        raise RuntimeError("Training did not produce a usable deepfake classifier head.")

    classifier.coef_ = best_coefficients
    classifier.intercept_ = best_intercept
    best_validation_metrics = evaluate_classifier(classifier, val_scaled, y_val)
    test_metrics = evaluate_classifier(classifier, test_scaled, y_test)

    payload = {
        "image_size": int(args.image_size),
        "embedding_dim": int(train_scaled.shape[1]),
        "scaler": {
            "mean": mean.round(10).tolist(),
            "scale": scale.round(10).tolist(),
        },
        "model": {
            "type": "vision_linear_head",
            "backbone_name": args.backbone_name,
            "weights": classifier.coef_[0].round(10).tolist(),
            "bias": float(np.round(classifier.intercept_[0], 10)),
        },
        "training_summary": {
            "status": "trained",
            "model_name": f"{args.backbone_name}-linear-head",
            "dataset_dir": str(dataset_dir),
            "dataset_rows": int(len(y_train) + len(y_val) + len(y_test)),
            "train_rows": int(len(y_train)),
            "validation_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "max_train_per_label": int(args.max_train_per_label),
            "backbone_name": args.backbone_name,
            "image_size": int(args.image_size),
            "best_epoch": int(best_epoch),
            "target_accuracy": float(args.target_accuracy),
            "target_confidence": float(args.target_confidence),
            "target_reached": bool(
                float(best_validation_metrics["accuracy"]) >= args.target_accuracy
                and float(best_validation_metrics["avg_confidence"]) >= args.target_confidence
            ),
            "history": history,
            "best_validation_metrics": best_validation_metrics,
            "accuracy": float(test_metrics["accuracy"]),
            "avg_confidence": float(test_metrics["avg_confidence"]),
            "precision_fake": float(test_metrics["precision_fake"]),
            "recall_fake": float(test_metrics["recall_fake"]),
            "f1_fake": float(test_metrics["f1_fake"]),
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report": test_metrics["classification_report"],
            "test_metrics": test_metrics,
            "system_profile": {
                "cpu": "Intel Core i7-8550U class",
                "ram_gb": 16,
                "recommended_reason": "Frozen vision backbone with a linear head to keep CPU load manageable.",
            },
        },
    }

    bundle_path = output_dir / "bundle.json"
    bundle_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["training_summary"], indent=2))
    print(f"Saved deepfake vision bundle to: {bundle_path}")


if __name__ == "__main__":
    main()
