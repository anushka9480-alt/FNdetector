from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.deepfake_detection import FEATURE_NAMES, extract_feature_vector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(description="Train a lightweight deepfake detector bundle.")
    parser.add_argument(
        "--dataset-dir",
        default=str(ROOT / "external" / "DeepFake-Detect" / "prepared_dataset"),
        help="Dataset root containing real/ and fake/ image folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "deployment" / "deepfake_model"),
        help="Where to save the lightweight model bundle.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of images reserved for evaluation.",
    )
    parser.add_argument(
        "--max-images-per-label",
        type=int,
        default=32,
        help="Maximum images to load from each label folder. Keeps training bounded on laptop CPUs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    return parser.parse_args()


def iter_labelled_images(dataset_dir: Path, max_images_per_label: int) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label_name, label_value in (("real", 0), ("fake", 1)):
        label_dir = dataset_dir / label_name
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing dataset folder: {label_dir}")
        label_images = [
            image_path for image_path in sorted(label_dir.iterdir())
            if image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
        if max_images_per_label > 0 and len(label_images) > max_images_per_label:
            label_images = random.sample(label_images, max_images_per_label)
        for image_path in label_images:
            if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            samples.append((image_path, label_value))
    if len(samples) < 8:
        raise ValueError("Dataset is too small. Expected at least 8 labelled images.")
    return samples


def load_dataset(dataset_dir: Path, max_images_per_label: int) -> tuple[np.ndarray, np.ndarray]:
    samples = iter_labelled_images(dataset_dir, max_images_per_label=max_images_per_label)
    features: list[np.ndarray] = []
    labels: list[int] = []
    random.shuffle(samples)

    for image_path, label in samples:
        image = Image.open(image_path).convert("RGB")
        features.append(extract_feature_vector(image))
        labels.append(label)

    return np.vstack(features), np.array(labels, dtype=np.int32)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels = load_dataset(
        dataset_dir,
        max_images_per_label=args.max_images_per_label,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0

    classifier = LogisticRegression(max_iter=2000, solver="liblinear", random_state=args.seed)
    classifier.fit((x_train - mean) / scale, y_train)

    test_predictions = classifier.predict((x_test - mean) / scale)
    accuracy = float(accuracy_score(y_test, test_predictions))
    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0,
        target_names=["real", "fake"],
    )

    payload = {
        "feature_names": FEATURE_NAMES,
        "scaler": {
            "mean": mean.round(10).tolist(),
            "scale": scale.round(10).tolist(),
        },
        "model": {
            "coefficients": classifier.coef_[0].round(10).tolist(),
            "intercept": float(np.round(classifier.intercept_[0], 10)),
        },
        "training_summary": {
            "status": "trained",
            "model_name": "lightweight-deepfake-linear",
            "dataset_dir": str(dataset_dir),
            "dataset_rows": int(len(labels)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "max_images_per_label": int(args.max_images_per_label),
            "accuracy": accuracy,
            "classification_report": report,
            "system_profile": {
                "cpu": "Intel Core i7-8550U class",
                "ram_gb": 16,
                "recommended_reason": "Conservative cap to avoid memory spikes on laptop-class CPUs.",
            },
        },
    }

    bundle_path = output_dir / "bundle.json"
    bundle_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["training_summary"], indent=2))
    print(f"Saved lightweight deepfake bundle to: {bundle_path}")


if __name__ == "__main__":
    main()
