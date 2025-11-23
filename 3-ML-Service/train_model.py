from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Determine base directory for saving the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def main():
    # Load dataset
    data = fetch_california_housing(as_frame=False)
    X, y = data.data, data.target

    # Split dataset
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(Xtr, ytr)

    # Ensure model directory exists and save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
