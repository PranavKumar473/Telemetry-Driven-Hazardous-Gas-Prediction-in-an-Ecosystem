import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

GAS_META = {
    "co_gt": {"label": "Carbon Monoxide (CO)", "short": "co", "unit": "mg/m³", "warn": 4.0, "crit": 6.0},
    "no2_gt": {"label": "Nitrogen Dioxide (NO₂)", "short": "no2", "unit": "µg/m³", "warn": 150.0, "crit": 200.0},
    "c6h6_gt": {"label": "Benzene (C₆H₆)", "short": "c6h6", "unit": "µg/m³", "warn": 15.0, "crit": 25.0},
}

st.set_page_config(
    page_title="Hazardous Gas Telemetry Monitor",
    layout="wide",
)

st.title("Telemetry-Driven Hazardous Gas Prediction")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    node_id = st.text_input("Sensor Node ID", value="eco_node_01")
    selected_gas = st.selectbox(
        "Target Gas",
        options=list(GAS_META.keys()),
        format_func=lambda k: GAS_META[k]["label"],
    )
    horizon = st.selectbox("Forecast Horizon", options=["1h", "3h", "6h"], index=0)
    auto_refresh = st.toggle("Auto-Refresh Live Stream", value=True)
    refresh_sec = st.slider("Refresh Interval (sec)", min_value=1, max_value=10, value=2)

    st.divider()
    if st.button("Reset Telemetry Database", type="secondary", use_container_width=True):
        try:
            requests.post(f"{API_BASE}/reset", timeout=3)
            st.success("Database cleared.")
            st.rerun()
        except Exception as exc:
            st.error(f"Reset failed: {exc}")


def fetch_json(endpoint: str, params: dict | None = None):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"Cannot reach FastAPI backend at {API_BASE}: {exc}")
        return None


tab_live, tab_bench = st.tabs(["Live Telemetry & Alerts", "Offline Model Benchmark"])

# ================= TAB 1: LIVE MONITORING =================
with tab_live:
    data = fetch_json("/history", params={"node_id": node_id, "limit": 120})

    if not data or not data["telemetry"]:
        st.info("Waiting for telemetry packets. Start `simulator\\producer.py` to stream data.")
    else:
        tel_df = pd.DataFrame(data["telemetry"])
        pred_df = pd.DataFrame(data["predictions"])
        alerts_df = pd.DataFrame(data["alerts"])

        meta = GAS_META[selected_gas]
        pred_col = f"{meta['short']}_pred_{horizon}"

        xgb_df = pred_df[pred_df["model_name"] == "XGBoost"] if not pred_df.empty else pd.DataFrame()
        lstm_df = pred_df[pred_df["model_name"] == "PyTorch_LSTM"] if not pred_df.empty else pd.DataFrame()

        latest_tel = tel_df.iloc[-1]
        latest_xgb = xgb_df.iloc[-1] if not xgb_df.empty else None
        latest_lstm = lstm_df.iloc[-1] if not lstm_df.empty else None

        # --- KPI Row ---
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            f"Current {meta['short'].upper()}",
            f"{latest_tel[selected_gas]:.2f} {meta['unit']}",
            delta=f"Mode: {latest_tel['fault_mode']}",
        )
        c2.metric(
            f"XGBoost ({horizon} ahead)",
            f"{latest_xgb[pred_col]:.2f} {meta['unit']}" if latest_xgb is not None else "Warming up...",
            delta=f"{latest_xgb['latency_ms']:.2f} ms" if latest_xgb is not None else None,
        )
        c3.metric(
            f"LSTM ({horizon} ahead)",
            f"{latest_lstm[pred_col]:.2f} {meta['unit']}" if latest_lstm is not None else "Warming up...",
            delta=f"{latest_lstm['latency_ms']:.2f} ms" if latest_lstm is not None else None,
        )
        is_anom = bool(latest_xgb["is_anomaly"]) if latest_xgb is not None else False
        c4.metric("Anomaly Detector", "ANOMALY" if is_anom else "NORMAL")
        c5.metric("Active Alerts (Recent)", len(alerts_df))

        # --- Live Time-Series Chart ---
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=tel_df["recorded_at"],
                y=