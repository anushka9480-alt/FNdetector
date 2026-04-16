from pathlib import Path
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news_detector.env import load_project_env  # noqa: E402
from fake_news_detector.prediction import (  # noqa: E402
    combine_news_text,
    get_model_snapshot,
    load_metrics_report,
    load_training_config,
    predict_text,
)

load_project_env(ROOT)

MODEL_DIR = ROOT / os.environ.get("MODEL_DIR", "deployment/model")

app = FastAPI(
    title="FN Detector API",
    version="1.0.0",
    description="Prediction API for the FN Detector fake-news classifier.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "routes": ["/health", "/metrics", "/predict"],
    }


@app.get("/health")
def health() -> dict:
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


@app.get("/metrics")
def metrics() -> dict:
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


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
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
