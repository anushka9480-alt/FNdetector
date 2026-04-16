from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import re

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
)


LABEL_MAP = {0: "fake", 1: "real"}
DEFAULT_TOKENIZER_NAME = "bert-base-uncased"


def load_tokenizer(model_dir: Path):
    try:
        return AutoTokenizer.from_pretrained(model_dir)
    except ValueError as exc:
        print(
            f"Falling back to tokenizer '{DEFAULT_TOKENIZER_NAME}' because the saved model "
            f"directory does not include a usable tokenizer: {exc}"
        )
        return AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)


def load_model(model_dir: Path):
    try:
        return AutoModelForSequenceClassification.from_pretrained(model_dir)
    except ValueError as exc:
        if "model_type" not in str(exc):
            raise
        print(
            f"Falling back to an explicit BERT config because '{model_dir}' "
            f"is missing model_type metadata: {exc}"
        )
        config = BertConfig.from_pretrained(model_dir, num_labels=2)
        return BertForSequenceClassification.from_pretrained(model_dir, config=config)


@dataclass(frozen=True)
class LoadedModelBundle:
    tokenizer: object
    model: object
    max_length: int
    model_name: str


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = str(value).replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", cleaned).strip()


def combine_news_text(title: str | None, text: str | None) -> str:
    normalized_title = normalize_text(title)
    normalized_text = normalize_text(text)
    if normalized_title and normalized_text:
        return f"{normalized_title}. {normalized_text}"
    return normalized_title or normalized_text


def _resolve_model_dir(model_dir: Path) -> Path:
    resolved = Path(model_dir).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Model directory does not exist: {resolved}")
    return resolved


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _load_model_bundle(model_dir_value: str) -> LoadedModelBundle:
    model_dir = _resolve_model_dir(Path(model_dir_value))
    training_config = _read_json(model_dir / "training_config.json")

    tokenizer = load_tokenizer(model_dir)
    model = load_model(model_dir)
    model.to(torch.device("cpu"))
    model.eval()

    return LoadedModelBundle(
        tokenizer=tokenizer,
        model=model,
        max_length=int(training_config.get("max_length", 192)),
        model_name=str(training_config.get("model_name", model_dir.name)),
    )


def load_training_config(model_dir: Path) -> dict:
    resolved_model_dir = _resolve_model_dir(model_dir)
    return _read_json(resolved_model_dir / "training_config.json")


def load_metrics_report(model_dir: Path) -> dict:
    resolved_model_dir = _resolve_model_dir(model_dir)
    return _read_json(resolved_model_dir / "metrics.json")


def get_model_snapshot(model_dir: Path) -> dict:
    resolved_model_dir = _resolve_model_dir(model_dir)
    training_config = load_training_config(resolved_model_dir)
    metrics_report = load_metrics_report(resolved_model_dir)
    history = metrics_report.get("history", [])
    latest_epoch = history[-1] if history else {}

    return {
        "model_dir": str(resolved_model_dir),
        "model_name": training_config.get("model_name"),
        "max_length": training_config.get("max_length"),
        "device": metrics_report.get("device", "cpu"),
        "train_rows": metrics_report.get("train_rows"),
        "validation_rows": metrics_report.get("validation_rows"),
        "test_rows": metrics_report.get("test_rows"),
        "latest_epoch": latest_epoch,
        "test_metrics": metrics_report.get("test_metrics", {}),
    }


def predict_text(model_dir: Path, text: str) -> dict:
    cleaned_text = normalize_text(text)
    if not cleaned_text:
        raise ValueError("Prediction text cannot be empty.")

    bundle = _load_model_bundle(str(_resolve_model_dir(model_dir)))
    encoded = bundle.tokenizer(
        cleaned_text,
        truncation=True,
        max_length=bundle.max_length,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = bundle.model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=1).squeeze(0)

    predicted_label = int(torch.argmax(probabilities).item())
    confidence = float(torch.max(probabilities).item())
    return {
        "prediction": LABEL_MAP[predicted_label],
        "predicted_label": predicted_label,
        "confidence": confidence,
        "model_name": bundle.model_name,
        "text_length": len(cleaned_text),
        "scores": {
            "fake": float(probabilities[0].item()),
            "real": float(probabilities[1].item()),
        },
    }
