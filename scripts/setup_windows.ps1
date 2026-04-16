$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-train.txt

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate it with: .\.venv\Scripts\Activate.ps1"
