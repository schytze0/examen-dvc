import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw/raw.csv"
RAW_DATA_DIR = "dataraw"
PROCESSED_DATA_DIR = "data/processed"

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def data_splitting():
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop("date", axis=1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["splitting"]["test_size"], 
                                                        random_state=params["splitting"]["random_state"])

    X_train.to_csv(f"{PROCESSED_DATA_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DATA_DIR}/y_test.csv", index=False)

    print("Splitted data saved in:", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    data_splitting()