"""Reusable prediction and rule-based maintenance recommendation helpers."""
from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "maintenance_model.joblib"

def load_model():
    if not MODEL_PATH.exists():
        from train_model import train
        train()
    return joblib.load(MODEL_PATH)

def recommendation(row, risk):
    flags = []
    if row.get("temperature_c", 0) >= 85: flags.append("High temperature: inspect cooling, lubrication, and thermal load.")
    if row.get("vibration_mm_s", 0) >= 7: flags.append("High vibration: inspect bearings, alignment, and mounting.")
    if row.get("pressure_bar", 99) <= 5: flags.append("Low pressure: inspect leaks, filters, and pump performance.")
    if row.get("hours_since_service", 0) >= 1000: flags.append("Service interval elevated: review planned-maintenance schedule.")
    if risk >= .7: flags.insert(0, "HIGH MODEL RISK: prioritize technician review and verify sensor readings.")
    elif risk >= .4: flags.insert(0, "MEDIUM MODEL RISK: schedule an inspection and monitor the next readings.")
    else: flags.insert(0, "LOW MODEL RISK: continue routine monitoring; this is not a guarantee of safe operation.")
    return flags

def predict_frame(frame):
    bundle = load_model(); X = frame[bundle["features"]]
    return bundle["model"].predict_proba(X)[:, 1]
