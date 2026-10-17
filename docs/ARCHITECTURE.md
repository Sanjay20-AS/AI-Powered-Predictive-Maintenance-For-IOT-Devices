# Architecture and design notes

```text
Virtual sensor generator
        │
        ▼
CSV telemetry + simulated failure label
        │
        ▼
Data split → preprocessing (imputation + one-hot encoding)
        │
        ▼
Random Forest classifier → held-out metrics + serialized pipeline
        │
        ▼
Streamlit scoring dashboard → fleet ranking / signal charts / CSV export
```

## Design choices
- **Dataset-first:** reproducible synthetic records mean no hardware is required.
- **Pipeline:** preprocessing is fit only on training data and reused at inference.
- **Class balancing:** helps the model pay attention to the minority failure class; still requires threshold and cost analysis.
- **Human-in-the-loop:** recommendations ask for inspection rather than triggering automatic shutdowns or work orders.
- **Observability:** saved metrics and downloadable scored telemetry make the demo inspectable.

## Production gap
Before any real deployment, validate against representative, time-aligned labeled asset data; use chronological/grouped evaluation; assess calibration and operating thresholds; monitor sensor quality and drift; secure ingestion and credentials; define alert ownership and audit logs; and have qualified maintenance personnel validate recommendations.
