import os
import pandas as pd
import yaml
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

PROCESSES_DATA_DIR = "data/processed"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

def grid_search():
    X_train = pd.read_csv(f"{PROCESSES_DATA_DIR}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{PROCESSES_DATA_DIR}/y_train.csv")

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    params_grid = {
        "n_estimators": params["grid_search"]["n_estimators"],
        "max_depth": params["grid_search"]["max_depth"],
        "min_samples_split": params["grid_search"]["min_samples_split"]
    }

    model = RandomForestRegressor(random_state=123)
    grid_search = GridSearchCV(model, params_grid, cv=5)
    grid_search.fit(X_train, y_train.values.squeeze())

    joblib.dump(grid_search.best_params_, f"{MODELS_DIR}/best_params.pkl")

    print("GridSearch finished. Saved best parameters to:", MODELS_DIR)

if __name__ == "__main__":
    grid_search()