from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import os
import re

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import joblib
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
)

from fake_news_detector.fact_check import analyze_fact_check_signals


LABEL_MAP = {0: "fake", 1: "real"}
DEFAULT_TOKENIZER_NAME = "bert-base-uncased"
SKLEARN_MODEL_FILENAME = "model.joblib"


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
    backend: str
    decision_threshold: float
    uncertainty_margin: float
    temperature: float


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
    metrics_report = _read_json(model_dir / "metrics.json")

    sklearn_model_path = model_dir / SKLEARN_MODEL_FILENAME
    if sklearn_model_path.exists():
        return LoadedModelBundle(
            tokenizer=None,
            model=joblib.load(sklearn_model_path),
            max_length=int(training_config.get("max_length", 0)),
            model_name=str(training_config.get("model_name", sklearn_model_path.stem)),
            backend="sklearn",
            decision_threshold=float(training_config.get("decision_threshold", 0.5)),
            uncertainty_margin=float(training_config.get("uncertainty_margin", 0.0)),
            temperature=float(metrics_report.get("calibration", {}).get("temperature", 1.0)),
        )

    tokenizer = load_tokenizer(model_dir)
    model = load_model(model_dir)
    model.to(torch.device("cpu"))
    model.eval()

    return LoadedModelBundle(
        tokenizer=tokenizer,
        model=model,
        max_length=int(training_config.get("max_length", 192)),
        model_name=str(training_config.get("model_name", model_dir.name)),
        backend="transformers",
        decision_threshold=float(training_config.get("decision_threshold", 0.65)),
        uncertainty_margin=float(training_config.get("uncertainty_margin", 0.1)),
        temperature=float(metrics_report.get("calibration", {}).get("temperature", 1.0)),
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
        "calibration": metrics_report.get("calibration", {}),
        "test_metrics": metrics_report.get("test_metrics", {}),
    }


def apply_temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(float(temperature), 1e-3)


def choose_verdict(scores: dict[str, float], threshold: float, uncertainty_margin: float) -> str:
    fake_score = float(scores["fake"])
    real_score = float(scores["real"])

    if fake_score >= threshold and (fake_score - real_score) >= uncertainty_margin:
        return "fake"
    if real_score >= threshold and (real_score - fake_score) >= uncertainty_margin:
        return "real"
    return "uncertain"


def apply_fact_check_guardrails(verdict: str, fact_check_signals: dict) -> str:
    if verdict != "real":
        return verdict

    high_risk = fact_check_signals.get("risk_level") == "high"
    low_trust = fact_check_signals.get("source_signal") == "low_trust_cues"
    weak_date_context = fact_check_signals.get("date_signal") in {"relative_date_only", "no_date_context"}
    has_trusted_source = bool(fact_check_signals.get("trusted_mentions"))

    if (high_risk or low_trust) and weak_date_context and not has_trusted_source:
        return "uncertain"
    return verdict


def predict_text(model_dir: Path, text: str) -> dict:
    cleaned_text = normalize_text(text)
    if not cleaned_text:
        raise ValueError("Prediction text cannot be empty.")

    bundle = _load_model_bundle(str(_resolve_model_dir(model_dir)))
    if bundle.backend == "sklearn":
        probabilities = bundle.model.predict_proba([cleaned_text])[0]
        class_to_probability = {
            int(label): float(probability)
            for label, probability in zip(bundle.model.classes_, probabilities)
        }
        fake_score = class_to_probability.get(0, 0.0)
        real_score = class_to_probability.get(1, 0.0)
        predicted_label = 0 if fake_score >= real_score else 1
        confidence = max(fake_score, real_score)
        return {
            "prediction": LABEL_MAP[predicted_label],
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "model_name": bundle.model_name,
            "text_length": len(cleaned_text),
            "scores": {
                "fake": float(fake_score),
                "real": float(real_score),
            },
        }

    encoded = bundle.tokenizer(
        cleaned_text,
        truncation=True,
        max_length=bundle.max_length,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = bundle.model(**encoded)
        scaled_logits = apply_temperature_scaling(outputs.logits, bundle.temperature)
        probabilities = torch.softmax(scaled_logits, dim=1).squeeze(0)

    predicted_label = int(torch.argmax(probabilities).item())
    confidence = float(torch.max(probabilities).item())
    scores = {
        "fake": float(probabilities[0].item()),
        "real": float(probabilities[1].item()),
    }
    verdict = choose_verdict(
        scores=scores,
        threshold=bundle.decision_threshold,
        uncertainty_margin=bundle.uncertainty_margin,
    )
    fact_check_signals = analyze_fact_check_signals(cleaned_text)
    verdict = apply_fact_check_guardrails(verdict, fact_check_signals)
    return {
        "prediction": verdict,
        "model_label": LABEL_MAP[predicted_label],
        "predicted_label": predicted_label,
        "confidence": confidence,
        "model_name": bundle.model_name,
        "text_length": len(cleaned_text),
        "scores": scores,
        "calibration": {
            "temperature": bundle.temperature,
            "threshold": bundle.decision_threshold,
            "uncertainty_margin": bundle.uncertainty_margin,
        },
        "fact_check_signals": fact_check_signals,
    }
