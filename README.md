# Data Project

## Dataset
Medical Cost Personal Dataset
https://www.kaggle.com/datasets/mirichoi0218/insurance

## Goal
Передбачити суму, яку страхова компанія виставить клієнту (charges), на основі його демографічних та фізичних показників.

## Setup
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

## Run
python run.py --config configs/config.yaml

## Outputs
- logs/run_<id>.log
- artifacts/run_<id>/
  - models/model.joblib
  - metrics/metrics.json
  - predictions/predictions.csv
  - reports/report.md
  - reports/config_snapshot.yaml
