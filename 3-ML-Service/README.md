# Model as a Service — ML Predictor

This project implements a simple Model-as-a-Service using Flask and Docker. It trains a RandomForest model on the California Housing dataset and provides a web UI and JSON API for predictions.

Quick start (without Docker)

1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train model

```bash
python train_model.py
```

3. Run app

```bash
python app.py
# then open http://127.0.0.1:5000/
```

Docker (build & run)

```bash
docker build -t ds-ml-service .
docker run --rm -p 5000:5000 ds-ml-service
```

API

- `GET /health` — returns `{ "ok": true, "features": [...] }`
- `POST /predict` — JSON body `{ "features": [ ...8 numbers... ] }` returns `{ "prediction": <value> }`

Notes

- The model is saved to `model/model.pkl` by `train_model.py`. The Dockerfile runs training during the image build.
- Do not commit large model files; see `.gitignore`.
