from pathlib import Path
import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.deepfake_detection import (  # noqa: E402
    get_deepfake_model_snapshot,
    predict_deepfake_image,
)
from fake_news_detector.prediction import (  # noqa: E402
    combine_news_text,
    get_model_snapshot,
    load_metrics_report,
    load_training_config,
    predict_text,
)


def _parse_cors_origins() -> list[str]:
    raw_value = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
    if not raw_value or raw_value == "*":
        return ["*"]
    return [item.strip() for item in raw_value.split(",") if item.strip()]


MODEL_DIR = ROOT / os.environ.get("MODEL_DIR", "deployment/model")
DEEPFAKE_MODEL_DIR = ROOT / os.environ.get("DEEPFAKE_MODEL_DIR", "deployment/deepfake_model")

app = FastAPI(
    title="FN Detector API",
    version="1.0.0",
    description="Prediction API for the FN Detector fake-news classifier.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    title: str = Field(default="", max_length=300)
    text: str = Field(default="", max_length=25000)


@app.get("/")
def root() -> dict:
    return {
        "service": "fn-detector-api",
        "status": "online",
        "model_dir": str(MODEL_DIR),
        "deepfake_model_dir": str(DEEPFAKE_MODEL_DIR),
        "routes": [
            "/health",
            "/metrics",
            "/predict",
            "/deepfake/health",
            "/deepfake/metrics",
            "/deepfake/predict",
            "/api/health",
            "/api/metrics",
            "/api/predict",
            "/api/deepfake/health",
            "/api/deepfake/metrics",
            "/api/deepfake/predict",
        ],
    }


def _news_health() -> dict:
    try:
        snapshot = get_model_snapshot(MODEL_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive for deployment only
        raise HTTPException(status_code=500, detail=f"Unable to load model: {exc}") from exc

    return {
        "status": "ok",
        "service": "fn-detector-api",
        "model": snapshot.get("model_name"),
        "device": snapshot.get("device", "cpu"),
        "max_length": snapshot.get("max_length"),
    }


@app.get("/health")
@app.get("/api/health")
def health() -> dict:
    return _news_health()


def _news_metrics() -> dict:
    try:
        return {
            "status": "ok",
            "snapshot": get_model_snapshot(MODEL_DIR),
            "training_config": load_training_config(MODEL_DIR),
            "report": load_metrics_report(MODEL_DIR),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive for deployment only
        raise HTTPException(status_code=500, detail=f"Unable to read metrics: {exc}") from exc


@app.get("/metrics")
@app.get("/api/metrics")
def metrics() -> dict:
    return _news_metrics()


def _news_predict(request: PredictionRequest) -> dict:
    combined_text = combine_news_text(request.title, request.text)
    if not combined_text:
        raise HTTPException(status_code=400, detail="Please provide a title or article body.")

    try:
        prediction = predict_text(model_dir=MODEL_DIR, text=combined_text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive for deployment only
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "status": "ok",
        "input": {
            "title": request.title,
            "text_length": len(request.text.strip()),
            "combined_length": len(combined_text),
        },
        "prediction": prediction,
    }


@app.post("/predict")
@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    return _news_predict(request)


def _deepfake_health() -> dict:
    try:
        snapshot = get_deepfake_model_snapshot(DEEPFAKE_MODEL_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "service": "fn-detector-api",
        "deepfake_model": snapshot["summary"].get("model_name"),
        "mode": snapshot["summary"].get("status"),
        "feature_count": len(snapshot["feature_names"]),
    }


@app.get("/deepfake/health")
@app.get("/api/deepfake/health")
def deepfake_health() -> dict:
    return _deepfake_health()


def _deepfake_metrics() -> dict:
    try:
        return {
            "status": "ok",
            "snapshot": get_deepfake_model_snapshot(DEEPFAKE_MODEL_DIR),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/deepfake/metrics")
@app.get("/api/deepfake/metrics")
def deepfake_metrics() -> dict:
    return _deepfake_metrics()


def _deepfake_predict(filename: str, payload: bytes) -> dict:
    try:
        prediction = predict_deepfake_image(
            model_dir=DEEPFAKE_MODEL_DIR,
            image_bytes=payload,
            filename=filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive for deployment only
        raise HTTPException(status_code=500, detail=f"Deepfake prediction failed: {exc}") from exc

    return {"status": "ok", "prediction": prediction}


@app.post("/deepfake/predict")
@app.post("/api/deepfake/predict")
async def deepfake_predict(file: UploadFile = File(...)) -> dict:
    return _deepfake_predict(file.filename or "upload", await file.read())
