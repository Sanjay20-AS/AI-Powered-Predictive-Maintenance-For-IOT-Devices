from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import pandas as pd
import streamlit as st
import plotly.express as px
from predict import load_model, recommendation

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="IoT Predictive Maintenance", page_icon="🛠️", layout="wide")
st.title("🛠️ AI-Powered Predictive Maintenance System")
st.caption("Virtual IoT telemetry • Machine-learning risk scoring • Explainable maintenance triage")
st.warning("DEMO ONLY: data is synthetic and generated for learning. Do not use these scores for real equipment or safety-critical decisions.")

@st.cache_data
def load_data():
    p = ROOT / "data" / "iot_sensor_data.csv"
    if not p.exists():
        from generate_data import generate
        return generate()
    return pd.read_csv(p)

@st.cache_resource
def model_bundle(): return load_model()

df = load_data()
try:
    bundle = model_bundle()
    features = bundle["features"]
    if "timestamp" in df: df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["risk_score"] = bundle["model"].predict_proba(df[features])[:, 1]
except Exception as e:
    st.error(f"Model unavailable. Run `python src/train_model.py` first. Details: {e}")
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Telemetry records", f"{len(df):,}")
c2.metric("Virtual machines", df.machine_id.nunique())
c3.metric("Observed simulated failures", f"{df.failure.sum():,}")
c4.metric("High-risk records (≥70%)", f"{(df.risk_score >= .7).sum():,}")

st.subheader("Machine fleet overview")
left,right = st.columns([1,2])
with left:
    machine = st.selectbox("Select machine", sorted(df.machine_id.unique()))
    threshold = st.slider("High-risk alert threshold", 0.1, 0.95, 0.70, 0.05)
    chosen = df[df.machine_id == machine].sort_values("timestamp")
    latest = chosen.iloc[-1]
    risk = float(latest.risk_score)
    st.metric("Latest model risk", f"{risk:.1%}")
    if risk >= threshold: st.error("Alert: review this machine")
    else: st.success("Below selected alert threshold")
    st.markdown("**Recommended checks**")
    for item in recommendation(latest, risk): st.write(f"- {item}")
with right:
    fig = px.scatter(df.sample(min(1500, len(df)), random_state=42), x="temperature_c", y="vibration_mm_s",
                     color="risk_score", hover_data=["machine_id", "hours_since_service"],
                     color_continuous_scale="RdYlGn_r", title="Temperature vs vibration, colored by model risk")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Risk ranking by machine")
summary = df.groupby("machine_id", as_index=False).agg(mean_risk=("risk_score","mean"), max_risk=("risk_score","max"),
    high_risk_records=("risk_score", lambda s: int((s >= threshold).sum())), records=("risk_score","size"))
summary = summary.sort_values("mean_risk", ascending=False)
st.plotly_chart(px.bar(summary, x="machine_id", y="mean_risk", color="mean_risk", color_continuous_scale="RdYlGn_r",
                      title="Average predicted risk by machine"), use_container_width=True)
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Telemetry explorer")
selected = st.multiselect("Signals", ["temperature_c","vibration_mm_s","pressure_bar","rpm","current_a","load_pct"],
                          default=["temperature_c","vibration_mm_s"])
if selected:
    trend = chosen.tail(300).melt(id_vars="timestamp", value_vars=selected, var_name="signal", value_name="value")
    st.plotly_chart(px.line(trend, x="timestamp", y="value", color="signal", title=f"Recent signals for {machine}"), use_container_width=True)

st.subheader("Download scored telemetry")
st.download_button("Download scored CSV", df.to_csv(index=False).encode("utf-8"), "scored_telemetry.csv", "text/csv")
with st.expander("Model evaluation"):
    p = ROOT / "models" / "metrics.json"
    if p.exists():
        import json
        metrics = json.loads(p.read_text())
        st.json({k:v for k,v in metrics.items() if k != "classification_report"})
    else: st.info("Run training to create evaluation metrics.")
