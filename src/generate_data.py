"""Generate a reproducible synthetic IoT predictive-maintenance dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "iot_sensor_data.csv"

def generate(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    machine_id = rng.choice([f"M-{i:03d}" for i in range(1, 13)], n)
    age = rng.integers(1, 120, n)
    load = np.clip(rng.normal(68, 18, n), 10, 120)
    ambient = rng.normal(25, 4, n)
    vibration = np.clip(rng.normal(3.2 + load/100*1.2 + age/120*1.4, 1.1, n), .1, 12)
    temperature = np.clip(ambient + 24 + load*.19 + age*.035 + rng.normal(0, 3, n), 25, 115)
    pressure = np.clip(rng.normal(7.2 - load*.008 - age*.002, .45, n), 3, 9)
    rpm = np.clip(rng.normal(1500 + load*4 - vibration*10, 100, n), 700, 2300)
    current = np.clip(rng.normal(8 + load*.075 + vibration*.18, 1.1, n), 2, 25)
    hours_since_service = rng.integers(0, 1500, n)
    # A transparent simulated failure mechanism with noise; not a real-world failure model.
    risk = (-8.0 + .075*(temperature-55) + .42*(vibration-4) + .055*(hours_since_service/10)
            + .025*(age-50) + .025*(load-65) + .25*(current-13) + rng.normal(0, 1.0, n))
    probability = 1/(1+np.exp(-risk))
    failure = rng.binomial(1, probability)
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "machine_id": machine_id, "machine_age_days": age, "load_pct": load.round(2),
        "ambient_temp_c": ambient.round(2), "temperature_c": temperature.round(2),
        "vibration_mm_s": vibration.round(3), "pressure_bar": pressure.round(3),
        "rpm": rpm.round(1), "current_a": current.round(2),
        "hours_since_service": hours_since_service, "failure": failure})
    DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA, index=False)
    print(f"Wrote {len(df):,} rows to {DATA}; failure rate={df.failure.mean():.1%}")
    return df

if __name__ == "__main__": generate()
