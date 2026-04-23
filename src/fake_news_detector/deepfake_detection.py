from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import models


MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_IMAGE_SIZE = 160
VISION_FEATURE_NAMES = tuple(f"embedding_{index:04d}" for index in range(576))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LEGACY_IMAGE_SIZE = 128
LEGACY_FEATURE_NAMES = (
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
)


@dataclass(frozen=True)
class DeepfakeModelBundle:
    model_type: str
    backbone_name: str
    image_size: int
    embedding_dim: int
    feature_names: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    classifier_weights: np.ndarray
    classifier_bias: float
    training_summary: dict


@dataclass(frozen=True)
class LoadedBackbone:
    model: torch.nn.Module
    embedding_dim: int


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
    except Exception as exc:  # pragma: no cover
        raise ValueError("Unable to decode the uploaded image.") from exc
    return image.convert("RGB")


def _preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    rgb = (rgb - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()
    return tensor


def _prepare_legacy_rgb_array(image: Image.Image) -> np.ndarray:
    resized = image.resize((LEGACY_IMAGE_SIZE, LEGACY_IMAGE_SIZE), Image.Resampling.BILINEAR)
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
    boundary_energy = float(np.mean(np.abs(vertical_boundaries))) + float(np.mean(np.abs(horizontal_boundaries)))
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


def extract_legacy_feature_dict(image: Image.Image) -> dict[str, float]:
    rgb = _prepare_legacy_rgb_array(image)
    rgb_uint8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    gray = _compute_gray(rgb)
    laplacian = _compute_laplacian(gray)
    gradients = np.concatenate([np.abs(np.diff(gray, axis=0)).ravel(), np.abs(np.diff(gray, axis=1)).ravel()])
    saturation = _compute_saturation(rgb)
    mirrored = np.flip(rgb, axis=1)
    jpeg_mean, jpeg_std = _compute_jpeg_residual(rgb_uint8)

    return {
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


def _create_backbone(backbone_name: str) -> LoadedBackbone:
    if backbone_name != "mobilenet_v3_small":
        raise ValueError(f"Unsupported deepfake backbone: {backbone_name}")

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    backbone = models.mobilenet_v3_small(weights=weights)
    backbone.classifier = torch.nn.Identity()
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return LoadedBackbone(model=backbone, embedding_dim=576)


@lru_cache(maxsize=2)
def load_backbone(backbone_name: str) -> LoadedBackbone:
    torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 4))))
    return _create_backbone(backbone_name)


def extract_embedding(image: Image.Image, *, backbone_name: str, image_size: int) -> np.ndarray:
    backbone = load_backbone(backbone_name)
    tensor = _preprocess_image(image, image_size).unsqueeze(0)
    with torch.no_grad():
        embedding = backbone.model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
    return embedding


def _default_training_summary(model_dir: Path) -> dict:
    return {
        "status": "unavailable",
        "model_name": "vision-backbone-unavailable",
        "model_dir": str(model_dir),
        "notes": "No trained deepfake vision bundle was found.",
    }


@lru_cache(maxsize=2)
def load_deepfake_bundle(model_dir_value: str) -> DeepfakeModelBundle | None:
    model_dir = _resolve_model_dir(Path(model_dir_value))
    bundle_path = model_dir / "bundle.json"
    if not bundle_path.exists():
        return None

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    model_payload = payload.get("model", {})
    scaler_payload = payload.get("scaler", {})
    feature_names = tuple(payload.get("feature_names", []))
    if not feature_names:
        feature_names = VISION_FEATURE_NAMES if model_payload.get("type") == "vision_linear_head" else LEGACY_FEATURE_NAMES
    weights = model_payload.get("weights")
    if weights is None:
        weights = model_payload.get("coefficients", [])
    bias = model_payload.get("bias")
    if bias is None:
        bias = model_payload.get("intercept", 0.0)
    return DeepfakeModelBundle(
        model_type=str(model_payload.get("type", "vision_linear_head")),
        backbone_name=str(model_payload.get("backbone_name", "mobilenet_v3_small")),
        image_size=int(payload.get("image_size", DEFAULT_IMAGE_SIZE)),
        embedding_dim=int(payload.get("embedding_dim", len(weights))),
        feature_names=feature_names,
        scaler_mean=np.array(scaler_payload.get("mean", []), dtype=np.float32),
        scaler_scale=np.array(scaler_payload.get("scale", []), dtype=np.float32),
        classifier_weights=np.array(weights, dtype=np.float32),
        classifier_bias=float(bias),
        training_summary=payload.get("training_summary", {}),
    )


def get_deepfake_model_snapshot(model_dir: Path) -> dict:
    resolved = _resolve_model_dir(model_dir)
    bundle = load_deepfake_bundle(str(resolved))
    summary = bundle.training_summary if bundle else _default_training_summary(resolved)
    feature_names = list(bundle.feature_names) if bundle else list(VISION_FEATURE_NAMES)

    return {
        "model_dir": str(resolved),
        "available": bundle is not None,
        "feature_names": feature_names,
        "summary": summary,
    }


def predict_deepfake_image(model_dir: Path, image_bytes: bytes, filename: str | None = None) -> dict:
    resolved_model_dir = _resolve_model_dir(model_dir)
    bundle = load_deepfake_bundle(str(resolved_model_dir))
    if bundle is None:
        raise FileNotFoundError(f"No deepfake model bundle found in: {resolved_model_dir}")

    image = _load_image(image_bytes)
    if bundle.model_type == "vision_linear_head":
        raw_vector = extract_embedding(
            image,
            backbone_name=bundle.backbone_name,
            image_size=bundle.image_size,
        )
    else:
        feature_dict = extract_legacy_feature_dict(image)
        raw_vector = np.array([feature_dict[name] for name in bundle.feature_names], dtype=np.float32)

    normalized = (raw_vector - bundle.scaler_mean) / np.where(bundle.scaler_scale == 0, 1.0, bundle.scaler_scale)
    logit = float(np.dot(normalized, bundle.classifier_weights) + bundle.classifier_bias)
    fake_score = _sigmoid(logit)
    real_score = 1.0 - fake_score

    top_indices = np.argsort(np.abs(normalized))[-4:][::-1]
    top_signals = [
        {"name": str(bundle.feature_names[int(index)]), "value": float(normalized[int(index)])}
        for index in top_indices
    ]

    return {
        "prediction": "fake" if fake_score >= 0.5 else "real",
        "confidence": float(max(fake_score, real_score)),
        "scores": {
            "fake": float(fake_score),
            "real": float(real_score),
        },
        "model_name": str(bundle.training_summary.get("model_name", "mobilenet-v3-small-linear-head")),
        "model_status": str(bundle.training_summary.get("status", "trained")),
        "model_metrics": bundle.training_summary.get("test_metrics", {}),
        "filename": filename or "upload",
        "image_size": {"width": image.width, "height": image.height},
        "features": {
            "embedding_dim": int(bundle.embedding_dim),
            "backbone": bundle.backbone_name if bundle.model_type == "vision_linear_head" else "legacy-artifact-features",
            "input_size": int(bundle.image_size),
        },
        "top_signals": top_signals,
    }
