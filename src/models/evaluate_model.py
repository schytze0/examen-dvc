import os
import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

DATA_DIR = "data"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
METRICS_DIR = "metrics"

os.makedirs(METRICS_DIR, exist_ok=True)

def evaluate_model():
    X_test = pd.read_csv(f"{PROCESSED_DATA_DIR}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{PROCESSED_DATA_DIR}/y_test.csv")
    model = joblib.load(f"{MODELS_DIR}/model.pkl")

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    with open(f"{METRICS_DIR}/scores.json", "w") as f:
        json.dump({"MSE": mse, "R2 SCORE": r2}, f)

    print("Save model performance metrics to:", METRICS_DIR)

    pd.DataFrame(y_pred, columns=["Prediction"]).to_csv(f"{DATA_DIR}/predictions.csv", index=False)

    print("Save predictions to:", DATA_DIR)

if __name__ == "__main__":
    evaluate_model()