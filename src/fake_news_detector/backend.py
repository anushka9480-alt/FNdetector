from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fake_news_detector.data import (
    create_full_splits,
    create_quick_splits,
    load_raw_dataset,
    summarize_splits,
    write_splits,
)
from fake_news_detector.env import load_project_env
from fake_news_detector.prediction import predict_text
from fake_news_detector.training import train_model
from fake_news_detector.workflow import build_training_config, build_workflow_summary


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_dir: str | None = None


class TrainingRequest(BaseModel):
    preset: str = Field(default="quick")
    num_epochs: int | None = Field(default=None, ge=1, le=10)
    output_dir: str | None = None


class WorkflowManager:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.models_dir = project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_path = self.models_dir / "training_jobs.json"
        self.active_model_path = self.models_dir / "active_model.json"
        self._lock = threading.Lock()
        self._jobs = self._load_jobs()

    def _load_jobs(self) -> dict[str, dict]:
        if not self.jobs_path.exists():
            return {}
        jobs = json.loads(self.jobs_path.read_text(encoding="utf-8"))
        updated = False
        for payload in jobs.values():
            if payload.get("status") in {"queued", "running"}:
                payload["status"] = "failed"
                payload["error"] = (
                    payload.get("error")
                    or "Training process was interrupted before completion."
                )
                payload["updated_at"] = now_iso()
                updated = True
        if updated:
            self.jobs_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        return jobs

    def _save_jobs(self) -> None:
        self.jobs_path.write_text(json.dumps(self._jobs, indent=2), encoding="utf-8")

    def _record_job(self, job_id: str, payload: dict) -> None:
        with self._lock:
            self._jobs[job_id] = payload
            self._save_jobs()

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda job: job["created_at"], reverse=True)

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return job

    def get_active_model_dir(self) -> Path | None:
        if self.active_model_path.exists():
            payload = json.loads(self.active_model_path.read_text(encoding="utf-8"))
            candidate = self.project_root / payload["relative_path"]
            if candidate.exists():
                return candidate

        for relative_path in ("models/laptop_cpu", "models/smoke_run", "models/full_run"):
            candidate = self.project_root / relative_path
            if (candidate / "training_config.json").exists():
                return candidate
        return None

    def set_active_model(self, model_dir: Path, *, preset: str | None = None) -> None:
        payload = {
            "relative_path": str(model_dir.relative_to(self.project_root)).replace("\\", "/"),
            "preset": preset,
            "updated_at": now_iso(),
        }
        self.active_model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_model_summary(self, model_dir: Path | None = None) -> dict:
        resolved = model_dir or self.get_active_model_dir()
        if not resolved:
            return {
                "available": False,
                "active_model": None,
            }

        metrics_path = resolved / "metrics.json"
        config_path = resolved / "training_config.json"
        active_payload = {}
        if self.active_model_path.exists():
            active_payload = json.loads(self.active_model_path.read_text(encoding="utf-8"))

        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

        return {
            "available": True,
            "active_model": {
                "path": str(resolved.relative_to(self.project_root)).replace("\\", "/"),
                "preset": active_payload.get("preset"),
                "updated_at": active_payload.get("updated_at"),
                "config": config,
                "metrics": metrics.get("test_metrics"),
                "rows": {
                    "train": metrics.get("train_rows"),
                    "validation": metrics.get("validation_rows"),
                    "test": metrics.get("test_rows"),
                },
            },
        }

    def workflow_summary(self) -> dict:
        return build_workflow_summary(self.project_root)

    def prepare_data(self) -> dict:
        raw_dir = self.project_root / "data" / "raw"
        fake_path = raw_dir / "fake_news.csv"
        true_path = raw_dir / "true_news.csv"
        if not fake_path.exists() or not true_path.exists():
            raise FileNotFoundError(
                "Raw CSV files are missing. Expected data/raw/fake_news.csv and data/raw/true_news.csv."
            )

        processed_dir = self.project_root / "data" / "processed"
        dataset = load_raw_dataset(fake_path=fake_path, true_path=true_path)
        full_splits = create_full_splits(dataset, seed=42)
        quick_splits = create_quick_splits(
            full_splits,
            seed=42,
            rows_per_label={"train": 3000, "val": 500, "test": 500},
        )

        write_splits(full_splits, processed_dir / "full")
        write_splits(quick_splits, processed_dir / "quick")

        summary = {
            "full": summarize_splits(full_splits),
            "quick": summarize_splits(quick_splits),
        }
        (processed_dir / "dataset_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary

    def start_training(self, request: TrainingRequest) -> dict:
        workflow = self.workflow_summary()
        preset_details = next(
            (preset for preset in workflow["presets"] if preset["key"] == request.preset),
            None,
        )
        if not preset_details:
            raise ValueError(f"Unknown preset: {request.preset}")
        if not preset_details["available"]:
            raise FileNotFoundError(
                f"Preset '{request.preset}' is not available yet. Prepare the dataset first."
            )

        job_id = uuid.uuid4().hex[:12]
        created_at = now_iso()
        config = build_training_config(
            self.project_root,
            request.preset,
            num_epochs=request.num_epochs,
            output_dir=request.output_dir,
        )
        job = {
            "job_id": job_id,
            "status": "queued",
            "preset": request.preset,
            "output_dir": config.output_dir,
            "num_epochs": config.num_epochs,
            "created_at": created_at,
            "updated_at": created_at,
            "error": None,
            "metrics": None,
        }
        self._record_job(job_id, job)

        worker = threading.Thread(
            target=self._run_training_job,
            args=(job_id, config, request.preset),
            daemon=True,
        )
        worker.start()
        return job

    def _run_training_job(self, job_id: str, config, preset: str) -> None:
        job = self.get_job(job_id)
        job["status"] = "running"
        job["updated_at"] = now_iso()
        self._record_job(job_id, job)

        try:
            metrics = train_model(config, project_root=self.project_root)
            model_dir = self.project_root / config.output_dir
            self.set_active_model(model_dir, preset=preset)

            job["status"] = "completed"
            job["metrics"] = metrics
            job["updated_at"] = now_iso()
            self._record_job(job_id, job)
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            job["updated_at"] = now_iso()
            self._record_job(job_id, job)


def create_app(project_root: Path) -> FastAPI:
    load_project_env(project_root)
    manager = WorkflowManager(project_root)
    cors_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
    app = FastAPI(title="Fake News Detector API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not cors_origins or cors_origins == "*" else [
            origin.strip() for origin in cors_origins.split(",") if origin.strip()
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        active_model = manager.get_active_model_dir()
        return {
            "status": "ok",
            "project_root": str(project_root),
            "active_model": str(active_model.relative_to(project_root)).replace("\\", "/")
            if active_model
            else None,
            "running_jobs": sum(1 for job in manager.list_jobs() if job["status"] == "running"),
        }

    @app.get("/api/workflow")
    def workflow() -> dict:
        return manager.workflow_summary()

    @app.post("/api/prepare-data")
    def prepare_data() -> dict:
        try:
            summary = manager.prepare_data()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "prepared", "summary": summary}

    @app.get("/api/model")
    def model() -> dict:
        return manager.get_model_summary()

    @app.post("/api/predict")
    def predict(request: PredictRequest) -> dict:
        model_dir = manager.get_active_model_dir()
        if request.model_dir:
            model_dir = project_root / request.model_dir
            if not model_dir.exists():
                raise HTTPException(status_code=400, detail="Requested model directory was not found.")
        if not model_dir:
            raise HTTPException(status_code=400, detail="No trained model is available yet.")
        return predict_text(model_dir=model_dir, text=request.text.strip())

    @app.get("/api/train")
    def list_training_jobs() -> dict:
        return {"jobs": manager.list_jobs()}

    @app.post("/api/train")
    def start_training(request: TrainingRequest) -> dict:
        try:
            job = manager.start_training(request)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return job

    @app.get("/api/train/{job_id}")
    def get_training_job(job_id: str) -> dict:
        try:
            return manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Training job not found.") from exc

    return app
