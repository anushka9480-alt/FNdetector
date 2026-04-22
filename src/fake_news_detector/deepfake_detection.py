from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = 128
MAX_IMAGE_BYTES = 8 * 1024 * 1024
FEATURE_NAMES = [
    "gray_mean",
    "gray_std",
    "laplacian_var",
    "gradient_mean",
    "gradient_std",
    "high_frequency_ratio",
    "blockiness",
    "jpeg_residual_mean",
    "jpeg_residual_std",
    "mirror_difference",
    "saturation_mean",
    "saturation_std",
]


@dataclass(frozen=True)
class DeepfakeModelBundle:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    training_summary: dict


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _resolve_model_dir(model_dir: Path) -> Path:
    resolved = Path(model_dir).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Deepfake model directory does not exist: {resolved}")
    return resolved


def _load_image(source: bytes) -> Image.Image:
    if not source:
        raise ValueError("No image content was provided.")
    if len(source) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large. Please upload an image under 8 MB.")
    try:
        image = Image.open(io.BytesIO(source))
    except Exception as exc:  # pragma: no cover - defensive for malformed uploads
        raise ValueError("Unable to decode the uploaded image.") from exc
    return image.convert("RGB")


def _prepare_rgb_array(image: Image.Image) -> np.ndarray:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _compute_gray(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb[..., :3], np.array([0.2989, 0.5870, 0.1140], dtype=np.float32))


def _compute_laplacian(gray: np.ndarray) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    return (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - (4.0 * center)
    )


def _compute_frequency_ratio(gray: np.ndarray) -> float:
    centered = gray - float(gray.mean())
    spectrum = np.abs(np.fft.rfft2(centered))
    if not np.any(spectrum):
        return 0.0

    height, width = gray.shape
    y_coords = np.fft.fftfreq(height)[:, None]
    x_coords = np.fft.rfftfreq(width)[None, :]
    radius = np.sqrt((y_coords**2) + (x_coords**2))
    high_mask = radius >= 0.18
    high_energy = float(spectrum[high_mask].sum())
    total_energy = float(spectrum.sum()) + 1e-8
    return high_energy / total_energy


def _compute_blockiness(gray: np.ndarray) -> float:
    vertical_boundaries = gray[:, 8::8] - gray[:, 7:-1:8]
    horizontal_boundaries = gray[8::8, :] - gray[7:-1:8, :]
    all_vertical = np.diff(gray, axis=1)
    all_horizontal = np.diff(gray, axis=0)
    boundary_energy = float(np.mean(np.abs(vertical_boundaries))) + float(
        np.mean(np.abs(horizontal_boundaries))
    )
    overall_energy = float(np.mean(np.abs(all_vertical))) + float(np.mean(np.abs(all_horizontal))) + 1e-8
    return boundary_energy / overall_energy


def _compute_jpeg_residual(rgb_uint8: np.ndarray) -> tuple[float, float]:
    image = Image.fromarray(rgb_uint8, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=72, optimize=True)
    recompressed = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    diff = np.abs(rgb_uint8.astype(np.float32) - np.asarray(recompressed, dtype=np.float32)) / 255.0
    return float(diff.mean()), float(diff.std())


def _compute_saturation(rgb: np.ndarray) -> np.ndarray:
    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)
    return channel_max - channel_min


def extract_feature_dict(image: Image.Image) -> dict[str, float]:
    rgb = _prepare_rgb_array(image)
    rgb_uint8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    gray = _compute_gray(rgb)
    laplacian = _compute_laplacian(gray)
    gradients = np.concatenate(
        [
            np.abs(np.diff(gray, axis=0)).ravel(),
            np.abs(np.diff(gray, axis=1)).ravel(),
        ]
    )
    saturation = _compute_saturation(rgb)
    mirrored = np.flip(rgb, axis=1)
    jpeg_mean, jpeg_std = _compute_jpeg_residual(rgb_uint8)

    values = {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "laplacian_var": float(laplacian.var()),
        "gradient_mean": float(gradients.mean()),
        "gradient_std": float(gradients.std()),
        "high_frequency_ratio": _compute_frequency_ratio(gray),
        "blockiness": _compute_blockiness(gray),
        "jpeg_residual_mean": jpeg_mean,
        "jpeg_residual_std": jpeg_std,
        "mirror_difference": float(np.mean(np.abs(rgb - mirrored))),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
    }
    return values


def extract_feature_vector(image: Image.Image) -> np.ndarray:
    feature_dict = extract_feature_dict(image)
    return np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float32)


def _default_training_summary(model_dir: Path) -> dict:
    return {
        "status": "heuristic",
        "model_name": "artifact-heuristic-baseline",
        "model_dir": str(model_dir),
        "notes": "No trained lightweight deepfake bundle was found, so the fallback heuristic is active.",
    }


@lru_cache(maxsize=2)
def load_deepfake_bundle(model_dir_value: str) -> DeepfakeModelBundle | None:
    model_dir = _resolve_model_dir(Path(model_dir_value))
    bundle_path = model_dir / "bundle.json"
    if not bundle_path.exists():
        return None

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    return DeepfakeModelBundle(
        feature_names=tuple(payload.get("feature_names", FEATURE_NAMES)),
        mean=np.array(payload["scaler"]["mean"], dtype=np.float32),
        scale=np.array(payload["scaler"]["scale"], dtype=np.float32),
        coefficients=np.array(payload["model"]["coefficients"], dtype=np.float32),
        intercept=float(payload["model"]["intercept"]),
        training_summary=payload.get("training_summary", {}),
    )


def get_deepfake_model_snapshot(model_dir: Path) -> dict:
    resolved = _resolve_model_dir(model_dir)
    bundle = load_deepfake_bundle(str(resolved))
    summary = bundle.training_summary if bundle else _default_training_summary(resolved)

    return {
        "model_dir": str(resolved),
        "available": bundle is not None,
        "feature_names": FEATURE_NAMES,
        "summary": summary,
    }


def _heuristic_prediction(features: dict[str, float]) -> float:
    raw_score = (
        2.8 * features["jpeg_residual_mean"]
        + 1.6 * features["high_frequency_ratio"]
        + 0.7 * max(features["blockiness"] - 1.0, 0.0)
        + 1.4 * features["mirror_difference"]
        + 8.0 * features["laplacian_var"]
        - 0.6 * features["saturation_mean"]
    )
    centered = (raw_score - 0.22) * 4.0
    return min(max(_sigmoid(centered), 0.02), 0.98)


def predict_deepfake_image(model_dir: Path, image_bytes: bytes, filename: str | None = None) -> dict:
    image = _load_image(image_bytes)
    features = extract_feature_dict(image)
    vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)

    bundle = load_deepfake_bundle(str(_resolve_model_dir(model_dir)))
    if bundle is None:
        fake_score = _heuristic_prediction(features)
        model_name = "artifact-heuristic-baseline"
        model_status = "heuristic"
    else:
        normalized = (vector - bundle.mean) / np.where(bundle.scale == 0, 1.0, bundle.scale)
        logit = float(np.dot(normalized, bundle.coefficients) + bundle.intercept)
        fake_score = _sigmoid(logit)
        model_name = str(bundle.training_summary.get("model_name", "lightweight-deepfake-linear"))
        model_status = "trained"

    real_score = 1.0 - fake_score
    top_feature_names = sorted(
        FEATURE_NAMES,
        key=lambda feature_name: abs(features[feature_name]),
        reverse=True,
    )[:4]

    return {
        "prediction": "fake" if fake_score >= 0.5 else "real",
        "confidence": float(max(fake_score, real_score)),
        "scores": {
            "fake": float(fake_score),
            "real": float(real_score),
        },
        "model_name": model_name,
        "model_status": model_status,
        "filename": filename or "upload",
        "image_size": {"width": image.width, "height": image.height},
        "features": features,
        "top_signals": [
            {"name": name, "value": float(features[name])}
            for name in top_feature_names
        ],
    }
