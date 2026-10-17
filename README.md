# AI-Powered Predictive Maintenance System for IoT Devices

A portfolio-ready learning project that simulates industrial IoT telemetry, trains a machine-learning classifier to estimate failure risk, and displays fleet health in a Streamlit dashboard. It is designed for students without access to industrial machines or sensors.

> **Important:** All telemetry and failure labels are synthetic. The model learns patterns from a deliberately simulated failure mechanism. This is a demonstration of an end-to-end workflow—not validated industrial predictive maintenance, not a remaining-useful-life model, and not suitable for operational or safety decisions.

## Features
- Reproducible virtual telemetry for 12 machines and 6,000 hourly records
- Temperature, vibration, pressure, RPM, current, load, machine age, and time since service
- Random Forest classifier with preprocessing pipeline and class balancing
- Held-out test evaluation: ROC-AUC, average precision, precision, recall, F1, confusion matrix
- Streamlit dashboard with fleet KPIs, machine risk ranking, signal exploration, alert threshold, and CSV download
- Rule-based inspection suggestions to help interpret the telemetry
- Basic automated test

## Quick start (Python 3.10–3.12 recommended)

### Windows PowerShell
```powershell
cd predictive-maintenance
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/generate_data.py
python src/train_model.py
streamlit run app.py
```

### macOS / Linux
```bash
cd predictive-maintenance
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/generate_data.py
python src/train_model.py
streamlit run app.py
```
Open the local URL printed by Streamlit (typically `http://localhost:8501`). The dataset and trained model are included for convenience; regenerate/retrain any time using the commands above.

## Project structure
```text
predictive-maintenance/
├── app.py                       # Streamlit dashboard
├── data/iot_sensor_data.csv     # Synthetic telemetry dataset
├── docs/                        # Architecture and GitHub guide
├── models/                      # Trained pipeline and evaluation metrics
├── src/generate_data.py         # Virtual sensor simulation
├── src/train_model.py           # Training/evaluation pipeline
├── src/predict.py               # Prediction and recommendation helpers
├── tests/test_predict.py        # Small automated test
├── requirements.txt
└── README.md
```

## Data dictionary
| Field | Meaning | Unit / type |
|---|---|---|
| timestamp | Simulated hourly observation timestamp | datetime |
| machine_id | Virtual asset identifier | category |
| machine_age_days | Age of virtual machine | days |
| load_pct | Operating load proxy | % (simulated) |
| ambient_temp_c | Ambient temperature | °C |
| temperature_c | Machine temperature | °C |
| vibration_mm_s | Vibration velocity proxy | mm/s |
| pressure_bar | System pressure proxy | bar |
| rpm | Rotational speed | revolutions/minute |
| current_a | Electrical current proxy | A |
| hours_since_service | Runtime since service | hours (simulated) |
| failure | Simulated binary target, 1 = failure event | 0/1 |

## How it works
1. `generate_data.py` produces correlated virtual sensor readings and a noisy simulated failure label.
2. `train_model.py` splits records into train/test sets using stratification, one-hot encodes machine IDs, imputes missing values, and fits a Random Forest.
3. Evaluation metrics are saved to `models/metrics.json`; the fitted pipeline is saved to `models/maintenance_model.joblib`.
4. `app.py` loads the model, scores telemetry, and presents risk and triage information.

## Interpreting the score
The score is the classifier's estimated probability for the synthetic label under this dataset's assumptions. It is **not** a calibrated real-world probability that a machine will fail in a specific time window. Alert thresholds are configurable for demonstration. In real deployments, threshold selection depends on failure prevalence, false-alarm cost, missed-failure cost, lead time, and technician capacity.

## Run tests
```bash
pytest -q
```

## Next improvements for a stronger portfolio
- Replace random row split with chronological or machine-grouped validation to avoid leakage in realistic temporal data.
- Compare Logistic Regression, Random Forest, and gradient boosting; tune thresholds on validation data.
- Add time-window features (rolling mean, standard deviation, trend, and lagged signals).
- Evaluate precision-recall curves and false alarms per machine-day.
- Add an anomaly-detection baseline and compare it to supervised classification.
- Ingest a public dataset such as NASA C-MAPSS (turbofan degradation / RUL) or a public predictive-maintenance dataset, checking its license and target definition before use.
- Add Docker, CI with GitHub Actions, model/data versioning, and a model card.
- For real IoT: MQTT ingestion, schema validation, message queue, time-series database, monitoring, alert routing, and human approval before maintenance action.

## Limitations
- The data-generating rules are invented and simplified.
- The row-level random split can overstate generalization when adjacent or same-machine observations are correlated.
- Binary failure classification is not the same as remaining useful life (RUL) estimation.
- The app is a local demo, not hardened for authentication, multi-user deployment, or production monitoring.

## License
MIT. See `LICENSE`.
