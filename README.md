# IRIS Homework Pipeline - Minimal

This repository contains a minimal Python project that generates a small Iris-like dataset, trains a scikit-learn model, and contains pytest unit tests for data validation and evaluation. It also includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs the tests on push and pull requests.

How to run locally

1. Create a virtual environment and activate it (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run tests:

```powershell
pytest -q
```
