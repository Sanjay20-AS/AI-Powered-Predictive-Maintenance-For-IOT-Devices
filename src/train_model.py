"""Train and evaluate a baseline predictive-maintenance classifier."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["machine_id", "machine_age_days", "load_pct", "ambient_temp_c", "temperature_c",
            "vibration_mm_s", "pressure_bar", "rpm", "current_a", "hours_since_service"]
TARGET = "failure"

def train():
    data_path = ROOT / "data" / "iot_sensor_data.csv"
    if not data_path.exists():
        from generate_data import generate
        generate()
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    numeric = [c for c in FEATURES if c != "machine_id"]
    preprocess = ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric),
                                    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["machine_id"])])
    model = Pipeline([("preprocess", preprocess), ("classifier", RandomForestClassifier(
        n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))])
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= .5).astype(int)
    metrics = {"test_rows": int(len(y_test)), "failure_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, prob)), "average_precision": float(average_precision_score(y_test, prob)),
        "precision_at_0_5": float(precision_score(y_test, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_test, pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True),
        "note": "Synthetic simulated data; metrics demonstrate workflow only, not production performance."}
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "threshold": .5}, ROOT / "models" / "maintenance_model.joblib")
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report"}, indent=2))
    return metrics

if __name__ == "__main__": train()
