from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.data import (  # noqa: E402
    create_full_splits,
    create_quick_splits,
    load_raw_dataset,
    normalize_raw_dataset,
    summarize_splits,
    write_splits,
)
from fake_news_detector.env import load_project_env  # noqa: E402


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Download and prepare the Kaggle fake news dataset.")
    parser.add_argument(
        "--dataset",
        default="aadyasingh55/fake-news-classification",
        help="Kaggle dataset slug.",
    )
    parser.add_argument(
        "--download-dir",
        default=str(ROOT / "external" / "kaggle" / "fake-news-classification"),
        help="Where the Kaggle dataset should be downloaded and extracted.",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(ROOT / "data" / "processed"),
        help="Output directory for normalized train/val/test CSV files.",
    )
    parser.add_argument(
        "--quick-train-per-label",
        type=int,
        default=3000,
        help="Rows per label for the quick training split.",
    )
    parser.add_argument(
        "--quick-eval-per-label",
        type=int,
        default=500,
        help="Rows per label for the quick validation and test splits.",
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


def download_dataset(dataset: str, download_dir: Path) -> list[Path]:
    api = get_kaggle_api()
    download_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset, path=str(download_dir), unzip=True, quiet=False)
    return sorted(download_dir.rglob("*.csv"))


def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, sep=";", engine="python")


def find_named_split(csv_paths: list[Path], *keywords: str) -> Path | None:
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    for path in csv_paths:
        normalized = path.name.lower()
        if all(keyword in normalized for keyword in lowered_keywords):
            return path
    return None


def load_or_split_dataset(csv_paths: list[Path]) -> tuple[dict[str, pd.DataFrame], dict]:
    if not csv_paths:
        raise FileNotFoundError("No CSV files were found after Kaggle download.")

    train_path = find_named_split(csv_paths, "train")
    val_path = find_named_split(csv_paths, "val") or find_named_split(csv_paths, "valid")
    test_path = find_named_split(csv_paths, "test")

    if train_path and val_path and test_path:
        full_splits = {
            "train": normalize_raw_dataset(load_csv_with_fallback(train_path)),
            "val": normalize_raw_dataset(load_csv_with_fallback(val_path)),
            "test": normalize_raw_dataset(load_csv_with_fallback(test_path)),
        }
        return full_splits, {
            "source_mode": "provided_splits",
            "train_csv": str(train_path),
            "val_csv": str(val_path),
            "test_csv": str(test_path),
        }

    largest_csv = max(csv_paths, key=lambda path: path.stat().st_size)
    dataset = normalize_raw_dataset(load_csv_with_fallback(largest_csv))
    full_splits = create_full_splits(dataset, seed=42)
    return full_splits, {
        "source_mode": "generated_splits",
        "source_csv": str(largest_csv),
    }


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    download_dir = Path(args.download_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()

    csv_paths = download_dataset(args.dataset, download_dir)
    full_splits, source_details = load_or_split_dataset(csv_paths)
    quick_splits = create_quick_splits(
        full_splits,
        seed=42,
        rows_per_label={
            "train": args.quick_train_per_label,
            "val": args.quick_eval_per_label,
            "test": args.quick_eval_per_label,
        },
    )

    write_splits(full_splits, processed_dir / "full")
    write_splits(quick_splits, processed_dir / "quick")

    summary = {
        "dataset": args.dataset,
        "download_dir": str(download_dir),
        **source_details,
        "full": summarize_splits(full_splits),
        "quick": summarize_splits(quick_splits),
    }
    summary_path = processed_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
