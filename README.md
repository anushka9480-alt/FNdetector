# FN Detector

FN Detector is a fake-news detection project built for practical local training and browser-friendly deployment. It combines a lightweight transformer training pipeline, a FastAPI backend workflow for prediction and training jobs, and a Vite frontend that talks directly to the backend.

## Live deployment

- Frontend: `https://fn-detector.vercel.app`
- Backend API: `https://Harman823-fndetector-backend.hf.space`
- Backend health check: `https://Harman823-fndetector-backend.hf.space/health`

## What is included

- Structured raw, processed, notebook, model, script, and source folders.
- Automated dataset preparation for `smoke`, `quick`, and local `full` splits.
- CPU-friendly transformer training with gradient accumulation.
- Prediction helpers for CLI use and backend inference.
- FastAPI backend endpoints for health, workflow, prediction, model metadata, and training jobs.
- Vite frontend under `frontend/` connected to the backend workflow.

## Project layout

```text
configs/                    training presets
data/
  archives/                 original zip files
  raw/                      source CSVs
  processed/                smoke, quick, and local full splits
models/                     local checkpoints and active model metadata
notebooks/                  original exploratory notebook
scripts/                    data prep, training, prediction, and local serving
src/fake_news_detector/     reusable training, backend, and inference code
frontend/                   Vite frontend for the detector UI
```

## Recommended setup on Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd frontend
npm install
```

## Shared environment file

The project now uses a shared root environment file:

- local config file: [`.env`](C:/Users/harma/OneDrive/Desktop/FN/FNdetector/.env)
- example template: [`.env.example`](C:/Users/harma/OneDrive/Desktop/FN/FNdetector/.env.example)

The backend reads the root `.env` automatically, and the Vite frontend is configured to read `VITE_*` variables from that same root file.

Current shared variables:

- `PORT`
- `MODEL_DIR`
- `CORS_ALLOW_ORIGINS`
- `VITE_API_BASE_URL`
- `HF_TOKEN`

The current deployed backend URL is:

```text
https://Harman823-fndetector-backend.hf.space
```

## Prepare the training data

```powershell
python scripts\prepare_data.py
```

This generates:

- `data/processed/full/*.csv`
- `data/processed/quick/*.csv`
- `data/processed/dataset_summary.json`

## Recommended training sequence

1. `smoke`: validate the full stack quickly.
2. `quick`: create a CPU-friendly deployment candidate.
3. `full`: run the longest local training pass when you want a stronger model.

Smoke training example:

```powershell
python scripts\train_model.py --train-file data\processed\smoke\train.csv --validation-file data\processed\smoke\val.csv --test-file data\processed\smoke\test.csv --output-dir models\smoke_run --num-epochs 1
```

Default quick training:

```powershell
python scripts\train_model.py --config configs\laptop_cpu.json
```

## Run the backend

```powershell
python scripts\serve_api.py
```

Backend endpoints:

- `GET /api/health`
- `GET /api/workflow`
- `GET /api/model`
- `POST /api/predict`
- `POST /api/train`
- `GET /api/train/{job_id}`
- `POST /api/prepare-data`

## Run the frontend

```powershell
cd frontend
npm run dev
```

The frontend now defaults to the Hugging Face backend Space through `VITE_API_BASE_URL`. For local-only backend work, point it back to `http://127.0.0.1:8000`.

## Build the frontend

```powershell
cd frontend
npm run build
```

## Vercel deployment shape

- Push the repository to GitHub.
- Import the repo into Vercel.
- Configure the frontend project root as `frontend`.
- Point the frontend at the deployed backend URL with `VITE_API_BASE_URL`.
- Current recommended backend: `https://Harman823-fndetector-backend.hf.space`
- Keep training local; use Vercel for app hosting and inference orchestration, not model training.

## Hugging Face backend deployment

The repository includes a Docker Space bundle builder for hosting the backend API on Hugging Face Spaces.

Build the Space bundle:

```powershell
python scripts\build_hf_space_bundle.py
```

Publish it to a Docker Space with a Hugging Face write token:

```powershell
$env:HF_TOKEN="your_hugging_face_write_token"
python scripts\publish_hf_space.py --space-id your-username/fndetector-backend
```

This uploads a minimal backend package from `dist/hf_space_backend` with:

- `GET /`
- `GET /health`
- `GET /metrics`
- `POST /predict`

## Notes

- `models/` stays ignored except for `.gitkeep`, so local checkpoints do not bloat Git.
- The first training run may download model files from Hugging Face.
- Full processed splits are kept local and excluded from Git where appropriate.

## Final submission checklist

- Confirm the frontend production site is live:
  - `https://fn-detector.vercel.app`
- Confirm the backend API is live:
  - `https://Harman823-fndetector-backend.hf.space/health`
- Confirm the GitHub repository is up to date:
  - `https://github.com/anushka9480-alt/FNdetector`
- Confirm `.env` is not committed and no tokens or secrets appear in Git history.
- Confirm the frontend is pointing to the deployed backend:
  - `VITE_API_BASE_URL=https://Harman823-fndetector-backend.hf.space`
- Confirm the live prediction flow works from the UI with a sample article.
- Confirm the backend returns:
  - `GET /health`
  - `GET /metrics`
  - `POST /predict`
- Confirm the README matches the current deployment shape:
  - Vercel for frontend
  - Hugging Face Space for backend
  - local machine for training

## Redeployment checklist

- Update the backend code or model bundle locally.
- If backend logic or model files changed, publish the backend again:

```powershell
python scripts\publish_hf_space.py --space-id Harman823/fndetector-backend
```

- If frontend code changed, redeploy the Vercel frontend:
  - push to GitHub if Vercel is linked to the repo
  - or run `vercel deploy --prod --yes`
- Recheck both live URLs after deployment:
  - `https://fn-detector.vercel.app`
  - `https://Harman823-fndetector-backend.hf.space/health`
- Run one final end-to-end prediction test from the live frontend.
