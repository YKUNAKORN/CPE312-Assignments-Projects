from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

# Load model at startup. If model file is missing, raise a clear error.
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")

model = joblib.load(MODEL_PATH)

FEATURES = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]

app = Flask(__name__, template_folder="templates")


@app.get("/health")
def health():
    # Return simple health and features info
    return {"ok": True, "features": FEATURES}


@app.post("/predict")
def predict():
    # Expect JSON body like {"features": [..8 values..]}
    data = request.get_json(silent=True) or {}
    x = data.get("features")
    if not x or len(x) != len(FEATURES):
        return jsonify(error=f"Expected {len(FEATURES)} features: {FEATURES}"), 400
    try:
        X = np.array([x], dtype=float)
    except Exception as e:
        return jsonify(error=f"Invalid feature values: {e}"), 400
    yhat = model.predict(X).tolist()[0]
    return jsonify(prediction=yhat)


@app.route("/", methods=["GET", "POST"])
def main():
    # Render form on GET. On POST, read form inputs and predict.
    if request.method == "GET":
        return render_template("main.html")
    vals = [float(request.form.get(f)) for f in FEATURES]
    pred = model.predict(np.array([vals])).tolist()[0]
    return render_template("main.html", prediction=pred)


if __name__ == "__main__":
    # Run development server when executed directly
    app.run(host="0.0.0.0", port=5001, debug=True)
