---
title: FN Detector Backend
emoji: 🤗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
startup_duration_timeout: 1h
suggested_hardware: cpu-basic
license: mit
short_description: FastAPI backend for the FN Detector fake-news classifier.
models:
  - prajjwal1/bert-mini
tags:
  - fastapi
  - docker
  - text-classification
  - fake-news-detection
---

# FN Detector Backend

This Hugging Face Space serves the FN Detector backend as a Docker-based FastAPI app.

## Endpoints

- `GET /`
- `GET /health`
- `GET /metrics`
- `POST /predict`

## Example request

```bash
curl -X POST "https://YOUR-SPACE.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d "{\"title\":\"Sample headline\",\"text\":\"Sample article body\"}"
```
