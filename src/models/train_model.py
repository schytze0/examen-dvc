import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

def train_model():
    X_train = pd.read_csv(f"{PROCESSED_DATA_DIR}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{PROCESSED_DATA_DIR}/y_train.csv")
    best_params = joblib.load(f"{MODELS_DIR}/best_params.pkl")

    model = RandomForestRegressor(**best_params, random_state=123)
    model.fit(X_train, y_train.values.squeeze())

    joblib.dump(model, f"{MODELS_DIR}/model.pkl")

    print("Model training finished. Save trained model to:", MODELS_DIR)

if __name__ == "__main__":
    train_model()