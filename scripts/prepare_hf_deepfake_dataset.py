from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import DownloadConfig, load_dataset  # noqa: E402
from fake_news_detector.env import load_project_env  # noqa: E402


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Download and prepare a Hugging Face deepfake dataset into real/ and fake/ folders."
    )
    parser.add_argument(
        "--dataset-id",
        default="insanescw/20K_real_and_deepfake_images",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "external" / "hf_deepfake" / "prepared_dataset"),
        help="Destination root containing real/ and fake/ subfolders.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / "external" / "hf_cache"),
        help="Datasets cache directory.",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="Name of the image column.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the label column.",
    )
    parser.add_argument(
        "--fake-label",
        default="0",
        help="Label value representing fake samples. Matches the chosen dataset.",
    )
    parser.add_argument(
        "--real-label",
        default="1",
        help="Label value representing real samples. Matches the chosen dataset.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=5000,
        help="Maximum images to export per label. Use 0 for no cap.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Optional Hugging Face token for higher rate limits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    return real_dir, fake_dir


def normalize_label(value: object) -> str:
    return str(value).strip().lower()


def resolve_image_suffix(image) -> str:
    extension = Path(getattr(image, "filename", "") or "").suffix.lower()
    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return extension
    return ".png"


def save_image(image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def main() -> None:
    load_project_env(ROOT)
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    real_dir, fake_dir = ensure_dirs(output_dir)
    download_config = DownloadConfig(max_retries=8)

    dataset = load_dataset(
        args.dataset_id,
        split=args.split,
        cache_dir=str(cache_dir),
        token=args.token or None,
        download_config=download_config,
    )

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    fake_label = normalize_label(args.fake_label)
    real_label = normalize_label(args.real_label)
    per_label_limit = args.max_per_label if args.max_per_label and args.max_per_label > 0 else None
    exported = {"real": 0, "fake": 0}
    skipped = 0

    for index in indices:
        row = dataset[int(index)]
        label_value = normalize_label(row.get(args.label_column))
        if label_value == fake_label:
            label_name = "fake"
            destination_dir = fake_dir
        elif label_value == real_label:
            label_name = "real"
            destination_dir = real_dir
        else:
            skipped += 1
            continue

        if per_label_limit is not None and exported[label_name] >= per_label_limit:
            continue

        image = row.get(args.image_column)
        if image is None:
            skipped += 1
            continue

        suffix = resolve_image_suffix(image)
        destination = destination_dir / f"{label_name}_{exported[label_name]:05d}{suffix}"
        save_image(image, destination)
        exported[label_name] += 1

        if per_label_limit is not None and all(count >= per_label_limit for count in exported.values()):
            break

    summary = {
        "status": "prepared",
        "dataset_id": args.dataset_id,
        "split": args.split,
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "image_column": args.image_column,
        "label_column": args.label_column,
        "label_mapping": {
            "fake": args.fake_label,
            "real": args.real_label,
        },
        "exported": exported,
        "skipped": skipped,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
