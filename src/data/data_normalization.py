import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

import joblib

PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def data_normalization():
    X_train = pd.read_csv(f"{PROCESSED_DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DATA_DIR}/X_test.csv")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled).to_csv(f"{PROCESSED_DATA_DIR}/X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled).to_csv(f"{PROCESSED_DATA_DIR}/X_test_scaled.csv", index=False)

    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

    print("Saved normalized data in:", PROCESSED_DATA_DIR)

if __name__ == "__main__":
    data_normalization()
