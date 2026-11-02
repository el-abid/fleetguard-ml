# FleetGuard: replacing an LLM agent with a classical ML model

A FastAPI micro-service that replaces an **"Agent LLM"** node inside an Abafusion workflow. Same JSON contract in, same JSON contract out, but no LLM call and no per-request cost.

## Why

The original workflow sent telematics data (fuel consumption, idle time, refuel events) to an LLM agent, which returned a severity level and a short written analysis for each trip.

That worked, but every trip paid for a model call, added latency, and produced slightly different wording for identical inputs.

This service keeps the exact same output contract `{"severity": int, "analyse": str}` while replacing the LLM with a small supervised model plus a deterministic text template.

## Results

| Metric | Value |
|---|---|
| Inference cost | **-80%** vs LLM calls |
| Accuracy (holdout) | **87.3%** |
| Macro F1 (holdout) | **87.3%** |
| Output determinism | 100% (identical input, identical output) |

Model comparison on a stratified 80/20 split:

| Model | Accuracy | Macro F1 |
|---|---|---|
| Decision Tree (depth 8) | 85.1% | 85.2% |
| **Random Forest (200 trees, depth 10)** | **87.3%** | **87.3%** |
| XGBoost (200 trees, depth 5) | 86.4% | 86.4% |

## How it works

```
Abafusion workflow
      |
      v
Detection_anomaly  --->  JSON payload (trip + consumption + anomalies)
                              |
                              v
                    POST /predict-severity
                              |
                    +---------+----------+
                    |                    |
              feature extraction    template analysis
              (19 features)         (no LLM)
                    |                    |
                    +---------+----------+
                              |
                              v
                  {"severity": 1|2|3, "analyse": "..."}
```

Severity levels: `1 = NORMAL`, `2 = SURVEILLANCE`, `3 = CRITIQUE`.

The model never sees raw text. Features are extracted from the payload and aligned with the training schema, so the service accepts the same payload the old agent received, including new fields.

## Data and target leakage

Dataset: 5,000 trips, 19 features, 3-class target.

One issue was found during the audit: `anomaly_count >= 2` mapped to `CRITIQUE` deterministically on 1,205 rows (24.1% of the dataset). A tree model would learn that single split instead of weighting the continuous variables.

Fix: for multi-anomaly trips, an intensity score is computed from how far each active anomaly exceeds its threshold, and the 8% lowest-intensity rows are moved from `CRITIQUE` to `SURVEILLANCE`. The business trend is preserved, the shortcut is broken.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/predict-severity \
  -H "Content-Type: application/json" \
  -d @example_payload.json
```

```json
{"severity": 3, "analyse": "Une surconsommation critique de 48.38 % a ete detectee..."}
```

Health check: `GET /health` returns `{"status": "ok"}`.

## Project structure

```
app.py                              FastAPI service, /predict-severity endpoint
fleetguard_predict.py               Feature extraction, model inference, analysis template
fleetguard_severity_model.pkl       Trained Random Forest + feature schema
FleetGuard_ML_training.ipynb        Full training pipeline and evaluation
requirements.txt
```

The `.pkl` bundles the model, the ordered feature columns and the categorical columns, so inference stays aligned with training.

## Limitations and next steps

- The analysis text is templated. It covers the three anomaly types present in the data, not free-form cases.
- No retraining pipeline yet: the model is static once exported.
- Next step would be drift monitoring on `deviation_percent` and `idle_ratio_percent`, plus a scheduled retrain.
