import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report)

DATA_PATH = os.path.join("data", "results.csv")
OUTPUT_MODEL_PATH = "model.bin"
SEED = 42
OPTIMAL_THRESHOLD = 0.41
C_PARAM = 1.0
MAX_ITER = 1000
FEATURES = ["rival", "condicion_independiente", "anio"]
TARGET = "target"


def extract_year(tournament_name: str):
    match = re.search(r"\d{4}", str(tournament_name))
    return int(match.group()) if match else None

def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    print(f"Loading data from '{data_path}'...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    df_cai = df[(df["local_team"] == "Independiente") | (df["visitor_team"] == "Independiente")].copy()

    df_cai["anio"] = df_cai["date_name"].apply(extract_year)
    df_cai = df_cai[df_cai["anio"].notna()].copy()
    df_cai["anio"] = df_cai["anio"].astype(int)
    df_cai = df_cai[df_cai["anio"] <= 2019].copy()

    df_cai["condicion_independiente"] = np.where(
        df_cai["local_team"] == "Independiente",
        "Local",
        "Visitante"
    )

    df_cai["rival"] = np.where(
        df_cai["local_team"] == "Independiente",
        df_cai["visitor_team"],
        df_cai["local_team"]
    )

    condiciones_independiente = [
        (df_cai["visitor_team"] == "Independiente") & (df_cai["local_result"] == df_cai["visitor_result"]),
        (df_cai["local_team"] == "Independiente") & (df_cai["local_result"] == df_cai["visitor_result"]),
        (df_cai["local_team"] == "Independiente") & (df_cai["local_result"] > df_cai["visitor_result"]),
        (df_cai["local_team"] == "Independiente") & (df_cai["local_result"] < df_cai["visitor_result"]),
        (df_cai["visitor_team"] == "Independiente") & (df_cai["visitor_result"] > df_cai["local_result"]),
        (df_cai["visitor_team"] == "Independiente") & (df_cai["visitor_result"] < df_cai["local_result"])
    ]
    resultados_independiente = [
        "Empate visitante",
        "Empate local",
        "Victoria de local",
        "Derrota de local",
        "Victoria de visitante",
        "Derrota de visitante"
    ]
    df_cai["victoria_independiente"] = np.select(
        condiciones_independiente,
        resultados_independiente,
        default="Empate"
    )

    df_cai[TARGET] = (
        df_cai["victoria_independiente"].isin(["Victoria de local", "Victoria de visitante"])
    ).astype(int)

    print(f"Data prepared successfully: {len(df_cai)} matches found.")
    print(f"Target distribution (Win vs Non-Win):\n{df_cai[TARGET].value_counts(normalize=True).to_dict()}")

    return df_cai


def evaluate_predictions(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_pred_prob >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, y_pred_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }


def train_model():
    df_cai = load_and_preprocess_data(DATA_PATH)

    df_train_full, df_test = train_test_split(df_cai, test_size=0.2, random_state=SEED)
    df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=SEED)

    y_train = df_train[TARGET].values
    y_val = df_val[TARGET].values
    y_test = df_test[TARGET].values

    print("\n[Step 1] Validating tuned model on validation set...")
    dv_val = DictVectorizer(sparse=False)
    X_train = dv_val.fit_transform(df_train[FEATURES].to_dict(orient="records"))
    X_val = dv_val.transform(df_val[FEATURES].to_dict(orient="records"))

    val_model = LogisticRegression(C=C_PARAM, max_iter=MAX_ITER, random_state=SEED)
    val_model.fit(X_train, y_train)

    val_preds_prob = val_model.predict_proba(X_val)[:, 1]
    val_metrics = evaluate_predictions(y_val, val_preds_prob, threshold=OPTIMAL_THRESHOLD)
    print(f"Validation Set Metrics (Threshold = {OPTIMAL_THRESHOLD}):")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n[Step 2] Training final model on df_train_full (80% of data)...")
    dv = DictVectorizer(sparse=False)
    X_full = dv.fit_transform(df_train_full[FEATURES].to_dict(orient="records"))
    y_full = df_train_full[TARGET].values

    X_test = dv.transform(df_test[FEATURES].to_dict(orient="records"))

    final_model = LogisticRegression(C=C_PARAM, max_iter=MAX_ITER, random_state=SEED)
    final_model.fit(X_full, y_full)

    test_preds_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_preds_prob, threshold=OPTIMAL_THRESHOLD)
    print(f"\nFinal Test Set Metrics (Threshold = {OPTIMAL_THRESHOLD}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    test_preds = (test_preds_prob >= OPTIMAL_THRESHOLD).astype(int)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=["No Victoria", "Victoria"]))

    print(f"\n[Step 4] Saving model and DictVectorizer to '{OUTPUT_MODEL_PATH}'...")
    with open(OUTPUT_MODEL_PATH, "wb") as f_out:
        pickle.dump((dv, final_model), f_out)

    print(f"Training completed successfully! Artifact saved to '{OUTPUT_MODEL_PATH}'.")


if __name__ == "__main__":
    train_model()
