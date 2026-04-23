from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.env import load_project_env  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
REAL_TOKENS = {"real", "original", "authentic", "genuine"}
FAKE_TOKENS = {"fake", "deepfake", "manipulated", "synthetic"}


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Download and prepare the Kaggle deepfake image dataset.")
    parser.add_argument(
        "--dataset",
        default="manjilkarki/deepfake-and-real-images",
        help="Kaggle dataset slug.",
    )
    parser.add_argument(
        "--download-dir",
        default=str(ROOT / "external" / "kaggle" / "deepfake-and-real-images"),
        help="Where the Kaggle dataset should be downloaded and extracted.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "external" / "kaggle_deepfake" / "prepared_dataset"),
        help="Output folder containing real/ and fake/ image subdirectories.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing prepared images before copying the new dataset.",
    )
    parser.add_argument(
        "--max-images-per-label",
        type=int,
        default=6000,
        help="Maximum images to prepare for each label. Use 0 to keep every image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling a capped dataset.",
    )
    return parser


def get_kaggle_api():
    load_project_env(ROOT)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - surfaced to CLI users
        raise RuntimeError(
            "The Kaggle package is not installed. Run `python -m pip install kaggle` "
            "or install requirements-train.txt first."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    return api


def download_dataset(dataset: str, download_dir: Path) -> None:
    api = get_kaggle_api()
    download_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset, path=str(download_dir), unzip=True, quiet=False)


def detect_label(image_path: Path) -> str | None:
    tokens = {part.lower() for part in image_path.parts}
    if tokens & FAKE_TOKENS:
        return "fake"
    if tokens & REAL_TOKENS:
        return "real"
    return None


def collect_labelled_images(download_dir: Path) -> dict[str, list[Path]]:
    labelled = {"real": [], "fake": []}
    for image_path in sorted(download_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = detect_label(image_path)
        if label:
            labelled[label].append(image_path)

    if not labelled["real"] or not labelled["fake"]:
        counts = Counter(detect_label(path) or "unlabelled" for path in download_dir.rglob("*") if path.is_file())
        raise RuntimeError(
            "Unable to locate both real and fake image folders in the extracted Kaggle dataset. "
            f"Detected labels: {dict(counts)}"
        )
    return labelled


def sample_images(
    labelled_images: dict[str, list[Path]],
    *,
    max_images_per_label: int,
    seed: int,
) -> dict[str, list[Path]]:
    if max_images_per_label <= 0:
        return labelled_images

    rng = random.Random(seed)
    sampled: dict[str, list[Path]] = {}
    for label, image_paths in labelled_images.items():
        if len(image_paths) <= max_images_per_label:
            sampled[label] = image_paths
        else:
            sampled[label] = sorted(rng.sample(image_paths, max_images_per_label))
    return sampled


def copy_images(labelled_images: dict[str, list[Path]], output_dir: Path, clean_output: bool) -> dict[str, int]:
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    counts: dict[str, int] = {}
    for label, image_paths in labelled_images.items():
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        counts[label] = 0

        for index, image_path in enumerate(image_paths, start=1):
            suffix = image_path.suffix.lower()
            target_name = f"{label}_{index:06d}{suffix}"
            shutil.copy2(image_path, label_dir / target_name)
            counts[label] += 1

    return counts


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    download_dir = Path(args.download_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    download_dataset(args.dataset, download_dir)
    labelled_images = collect_labelled_images(download_dir)
    labelled_images = sample_images(
        labelled_images,
        max_images_per_label=args.max_images_per_label,
        seed=args.seed,
    )
    counts = copy_images(labelled_images, output_dir, clean_output=args.clean_output)

    summary = {
        "dataset": args.dataset,
        "download_dir": str(download_dir),
        "output_dir": str(output_dir),
        "max_images_per_label": int(args.max_images_per_label),
        "real_images": counts.get("real", 0),
        "fake_images": counts.get("fake", 0),
        "total_images": sum(counts.values()),
    }
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
