from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.data import (  # noqa: E402
    DEFAULT_HF_DATASET_CSV_URL,
    create_full_splits,
    create_quick_splits,
    load_huggingface_dataset_csv,
    summarize_splits,
    write_splits,
)


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_huggingface_dataset_csv()
    dataset_source = DEFAULT_HF_DATASET_CSV_URL
    dataset.to_csv(raw_dir / "FakeNewsNet_prepared.csv", index=False)

    full_splits = create_full_splits(dataset, seed=42)
    quick_splits = create_quick_splits(
        full_splits,
        seed=42,
        rows_per_label={"train": 3000, "val": 500, "test": 500},
    )

    write_splits(full_splits, processed_dir / "full")
    write_splits(quick_splits, processed_dir / "quick")

    summary = {
        "dataset_source": dataset_source,
        "dataset_rows": int(len(dataset)),
        "full": summarize_splits(full_splits),
        "quick": summarize_splits(quick_splits),
    }

    summary_path = processed_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
