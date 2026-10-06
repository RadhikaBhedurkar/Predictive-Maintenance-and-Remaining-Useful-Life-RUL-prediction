import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Predictive Maintenance & RUL Prediction")
st.caption("NASA C-MAPSS FD001 | End-to-end Machine Learning Demo")

try:
    model = joblib.load("artifacts/model.pkl")
    feature_columns = joblib.load("artifacts/feature_columns.pkl")
except FileNotFoundError:
    st.error("Artifacts are missing. Run: python -m src.pipeline.training_pipeline")
    st.stop()

defaults = {
    "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0,
    "sensor_2": 642.0, "sensor_3": 1589.0, "sensor_4": 1400.0,
    "sensor_7": 553.0, "sensor_8": 2388.0, "sensor_9": 9045.0,
    "sensor_11": 47.0, "sensor_12": 522.0, "sensor_13": 2388.0,
    "sensor_14": 8130.0, "sensor_15": 8.5, "sensor_17": 392.0,
    "sensor_20": 39.0, "sensor_21": 23.0
}

st.subheader("Machine sensor input")
values = {}
columns = st.columns(3)

for i, feature in enumerate(defaults):
    with columns[i % 3]:
        values[feature] = st.number_input(
            feature, value=float(defaults[feature])
        )

if st.button("Predict RUL", type="primary"):
    row = {}

    for feature in feature_columns:
        if feature in values:
            row[feature] = values[feature]
        elif feature.endswith("_rolling_mean"):
            base = feature.replace("_rolling_mean", "")
            row[feature] = values.get(base, 0.0)
        elif feature.endswith("_rolling_std"):
            row[feature] = 0.0
        else:
            row[feature] = 0.0

    X = pd.DataFrame([row], columns=feature_columns)
    rul = max(0.0, float(model.predict(X)[0]))

    if rul < 20:
        risk = "HIGH"
        action = "Schedule maintenance immediately."
    elif rul < 50:
        risk = "MEDIUM"
        action = "Plan maintenance soon and monitor the machine."
    else:
        risk = "LOW"
        action = "Continue normal monitoring."

    a, b, c = st.columns(3)
    a.metric("Predicted RUL", f"{rul:.1f} cycles")
    b.metric("Risk Level", risk)
    c.metric("Model", type(model).__name__)

    st.info(action)
