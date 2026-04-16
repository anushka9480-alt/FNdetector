from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_text(value: str) -> str:
    if pd.isna(value):
        return ""
    cleaned = str(value).replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", cleaned).strip()


def build_combined_text(title: str, text: str) -> str:
    title = normalize_text(title)
    text = normalize_text(text)
    if title and text:
        return f"{title}. {text}"
    return title or text


def load_raw_dataset(fake_path: Path, true_path: Path) -> pd.DataFrame:
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1
    fake_df["label_name"] = "fake"
    true_df["label"] = 0
    true_df["label_name"] = "real"

    combined = pd.concat([fake_df, true_df], ignore_index=True)
    combined["title"] = combined["title"].map(normalize_text)
    combined["text"] = combined["text"].map(normalize_text)
    combined["subject"] = combined["subject"].map(normalize_text)
    combined["date"] = combined["date"].map(normalize_text)
    combined["combined_text"] = [
        build_combined_text(title, text)
        for title, text in zip(combined["title"], combined["text"])
    ]

    combined = combined.loc[combined["combined_text"].str.len() > 0].copy()
    combined = combined[
        ["title", "text", "combined_text", "subject", "date", "label", "label_name"]
    ].reset_index(drop=True)
    return combined


def create_full_splits(dataset: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=seed,
    )

    return {
        "train": train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        "val": val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        "test": test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True),
    }


def create_quick_splits(
    full_splits: dict[str, pd.DataFrame],
    seed: int = 42,
    rows_per_label: dict[str, int] | None = None,
) -> dict[str, pd.DataFrame]:
    limits = rows_per_label or {"train": 3000, "val": 500, "test": 500}
    quick_splits: dict[str, pd.DataFrame] = {}

    for split_name, frame in full_splits.items():
        per_label = limits.get(split_name, 500)
        pieces = []
        for label in sorted(frame["label"].unique()):
            part = frame.loc[frame["label"] == label]
            pieces.append(part.sample(n=min(len(part), per_label), random_state=seed))

        sampled = pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed)
        sampled = sampled.reset_index(drop=True)
        quick_splits[split_name] = sampled

    return quick_splits


def write_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame in splits.items():
        frame.to_csv(output_dir / f"{split_name}.csv", index=False)


def summarize_splits(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split_name, frame in splits.items():
        label_counts = frame["label"].value_counts().sort_index().to_dict()
        summary[split_name] = {
            "rows": int(len(frame)),
            "real": int(label_counts.get(0, 0)),
            "fake": int(label_counts.get(1, 0)),
        }
    return summary


def load_split_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
