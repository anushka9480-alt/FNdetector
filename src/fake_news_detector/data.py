import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_HF_DATASET_CSV_URL = (
    "https://huggingface.co/datasets/rickstello/FakeNewsNet/resolve/main/FakeNewsNet.csv"
)


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


def _infer_schema(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(c).lower() for c in normalized.columns]

    title_series = normalized.get("title", pd.Series([""] * len(normalized), index=normalized.index))
    if "text" in normalized:
        text_series = normalized["text"]
    else:
        fallback_parts = [
            normalized.get("source_domain", pd.Series([""] * len(normalized), index=normalized.index)),
            normalized.get("news_url", pd.Series([""] * len(normalized), index=normalized.index)),
        ]
        text_series = pd.Series(
            [
                " ".join(normalize_text(value) for value in values if normalize_text(value))
                for values in zip(*fallback_parts)
            ],
            index=normalized.index,
        )

    if "label" in normalized:
        label_series = pd.to_numeric(normalized["label"], errors="coerce")
    elif "real" in normalized:
        real_series = pd.to_numeric(normalized["real"], errors="coerce")
        # Preserve the repository-wide label convention used by prediction.py:
        # 0 = fake, 1 = real.
        label_series = real_series
    else:
        raise ValueError(
            "Unsupported dataset schema. Expected either 'label' or 'real' column in the source data."
        )

    prepared = pd.DataFrame(
        {
            "title": title_series.fillna("").map(normalize_text),
            "text": text_series.fillna("").map(normalize_text),
            "label": label_series,
        }
    )
    prepared = prepared.dropna(subset=["label"]).copy()
    prepared["label"] = prepared["label"].astype(int)
    prepared = prepared.loc[prepared["label"].isin([0, 1])].copy()
    prepared["label_name"] = prepared["label"].map({0: "fake", 1: "real"})
    prepared["combined_text"] = [
        build_combined_text(title, text)
        for title, text in zip(prepared["title"], prepared["text"])
    ]
    prepared = prepared.loc[prepared["combined_text"].str.len() > 0].copy()
    return prepared[["title", "text", "combined_text", "label", "label_name"]].reset_index(drop=True)


def load_raw_dataset(dataset_path: Path) -> pd.DataFrame:
    combined = pd.read_csv(dataset_path)
    return _infer_schema(combined)


def load_huggingface_dataset_csv(dataset_csv_url: str = DEFAULT_HF_DATASET_CSV_URL) -> pd.DataFrame:
    combined = pd.read_csv(dataset_csv_url)
    return _infer_schema(combined)


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
            "fake": int(label_counts.get(0, 0)),
            "real": int(label_counts.get(1, 0)),
        }
    return summary


def load_split_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
