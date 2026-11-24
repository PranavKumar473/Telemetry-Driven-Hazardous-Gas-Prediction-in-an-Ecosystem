"""Streamlit Live Monitoring & Multi-Model Benchmark Dashboard."""
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from ml.config import MODELS_DIR

API_BASE = "http://127.0.0.1:8000"

GAS_META = {
    "co_gt": {
        "label": "Carbon Monoxide (CO)",
        "short": "co",
        "unit": "mg/m³",
        "elevated": 2.5,
        "warn": 4.0,
        "crit": 6.0,
    },
    "no2_gt": {
        "label": "Nitrogen Dioxide (NO₂)",
        "short": "no2",
        "unit": "µg/m³",
        "elevated": 100.0,
        "warn": 150.0,
        "crit": 200.0,
    },
    "c6h6_gt": {
        "label": "Benzene (C₆H₆)",
        "short": "c6h6",
        "unit": "µg/m³",
        "elevated": 10.0,
        "warn": 15.0,
        "crit": 25.0,
    },
}

SEVERITY_BADGE = {
    "CRITICAL": "🔴 CRITICAL",
    "WARNING": "🟠 WARNING",
    "ELEVATED": "🟡 ELEVATED",
    "SAFE": "🟢 SAFE",
}

st.set_page_config(
    page_title="Hazardous Gas Telemetry Monitor",
    layout="wide",
)

st.title("Telemetry-Driven Hazardous Gas Prediction in an Ecosystem")
st.caption(
    "Real-Time Multivariate Gas Forecasting (**XGBoost**, **PyTorch LSTM**, **1D-CNN+LSTM**) "
    "& Unsupervised Sensor Anomaly Detection (**Isolation Forest**)"
)


def fetch_json(endpoint: str, params: dict | None = None, silent: bool = False):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        if not silent:
            st.error(f"Cannot reach FastAPI backend at {API_BASE}: {exc}")
        return None


# --- Sidebar Controls ---
with st.sidebar:
    st.header("System Status")
    health = fetch_json("/health", silent=True)
    if health and health.get("status") == "ok":
        st.success(f"🟢 Backend Online (SEQ_LEN={health.get('seq_len', 24)}h)")
        st.caption("Loaded: " + ", ".join(health.get("loaded_models", [])))
    else:
        st.error("🔴 Backend Offline")
        st.caption("Start with: `uvicorn backend.main:app --reload`")

    st.divider()
    st.header("Telemetry Controls")
    node_id = st.text_input("Sensor Node ID", value="eco_node_01")
    selected_gas = st.selectbox(
        "Target Gas",
        options=list(GAS_META.keys()),
        format_func=lambda k: GAS_META[k]["label"],
    )
    horizon = st.selectbox("Forecast Horizon", options=["1h", "3h", "6h"], index=0)
    auto_refresh = st.toggle("Auto-Refresh Live Stream", value=False)
    refresh_sec = st.slider("Refresh Interval (sec)", min_value=1, max_value=10, value=2)

    st.divider()
    if st.button("Reset Telemetry Database", type="secondary", use_container_width=True):
        try:
            requests.post(f"{API_BASE}/reset", timeout=3).raise_for_status()
            st.success("Database and alert cooldowns cleared.")
            st.rerun()
        except Exception as exc:
            st.error(f"Reset failed: {exc}")


tab_live, tab_bench = st.tabs(["Live Telemetry & Multi-Model Forecasts", "Offline Model Benchmark & Loss Curves"])

# ================= TAB 1: LIVE MONITORING =================
with tab_live:
    data = fetch_json("/history", params={"node_id": node_id, "limit": 120}, silent=True)

    if not data or not data.get("telemetry"):
        st.info(
            "Waiting for live telemetry packets. Run `python -m simulator.producer --mode leak --interval 0.5 --limit 60` "
            "in a terminal to stream packets into FastAPI."
        )
    else:
        tel_df = pd.DataFrame(data["telemetry"])
        pred_df = pd.DataFrame(data["predictions"])
        alerts_df = pd.DataFrame(data["alerts"])

        meta = GAS_META[selected_gas]
        pred_col = f"{meta['short']}_pred_{horizon}"

        xgb_df = pred_df[pred_df["model_name"] == "XGBoost"] if not pred_df.empty else pd.DataFrame()
        lstm_df = pred_df[pred_df["model_name"] == "PyTorch_LSTM"] if not pred_df.empty else pd.DataFrame()
        cnn_df = pred_df[pred_df["model_name"] == "CNN_LSTM"] if not pred_df.empty else pd.DataFrame()

        latest_tel = tel_df.iloc[-1]
        latest_xgb = xgb_df.iloc[-1] if not xgb_df.empty else None
        latest_lstm = lstm_df.iloc[-1] if not lstm_df.empty else None
        latest_cnn = cnn_df.iloc[-1] if not cnn_df.empty else None

        # --- 6-Column KPI Row ---
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric(
            f"Actual {meta['short'].upper()}",
            f"{latest_tel[selected_gas]:.2f} {meta['unit']}",
            delta=f"Mode: {latest_tel['fault_mode']} (#{len(tel_df)})",
        )
        c2.metric(
            f"XGBoost (+{horizon})",
            f"{latest_xgb[pred_col]:.2f} {meta['unit']}" if latest_xgb is not None else f"Warm ({len(tel_df)}/7)",
            delta=f"{latest_xgb['latency_ms']:.2f} ms" if latest_xgb is not None else None,
        )
        c3.metric(
            f"PyTorch LSTM (+{horizon})",
            f"{latest_lstm[pred_col]:.2f} {meta['unit']}" if latest_lstm is not None else f"Warm ({len(tel_df)}/24)",
            delta=f"{latest_lstm['latency_ms']:.2f} ms" if latest_lstm is not None else None,
        )
        c4.metric(
            f"1D-CNN+LSTM (+{horizon})",
            f"{latest_cnn[pred_col]:.2f} {meta['unit']}" if latest_cnn is not None else f"Warm ({len(tel_df)}/24)",
            delta=f"{latest_cnn['latency_ms']:.2f} ms" if latest_cnn is not None else None,
        )
        is_anom = bool(latest_xgb["is_anomaly"]) if latest_xgb is not None else False
        c5.metric("Isolation Forest", "🟠 ANOMALY" if is_anom else "🟢 NORMAL")
        c6.metric("Triggered Alerts", len(alerts_df))

        # --- Live Time-Series Chart ---
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=tel_df["recorded_at"],
                y=tel_df[selected_gas],
                mode="lines+markers",
                name=f"Actual {meta['label']}",
                line={"color": "#1f77b4", "width": 2.5},
            )
        )
        if not xgb_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=xgb_df["recorded_at"],
                    y=xgb_df[pred_col],
                    mode="lines",
                    name=f"XGBoost Forecast (+{horizon})",
                    line={"color": "#ff7f0e", "width": 2, "dash": "dot"},
                )
            )
        if not lstm_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=lstm_df["recorded_at"],
                    y=lstm_df[pred_col],
                    mode="lines",
                    name=f"PyTorch LSTM Forecast (+{horizon})",
                    line={"color": "#2ca02c", "width": 2},
                )
            )
        if not cnn_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=cnn_df["recorded_at"],
                    y=cnn_df[pred_col],
                    mode="lines",
                    name=f"1D-CNN+LSTM Forecast (+{horizon})",
                    line={"color": "#9467bd", "width": 2.5},
                )
            )

        fig.add_hline(
            y=meta["warn"],
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Warning ({meta['warn']} {meta['unit']})",
            annotation_position="top left",
        )
        fig.add_hline(
            y=meta["crit"],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Critical ({meta['crit']} {meta['unit']})",
            annotation_position="top left",
        )
        fig.update_layout(
            title=f"{meta['label']} — Live Stream vs. +{horizon} Multi-Model Forecasts (Node: {node_id})",
            xaxis_title="Timestamp",
            yaxis_title=f"Concentration ({meta['unit']})",
            height=430,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            margin={"l": 40, "r": 20, "t": 60, "b": 40},
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Split Tiered Alert Feed (Gas Breaches vs. Sensor Hardware Faults) ---
        st.subheader("Tiered Incident & Alert Feed (Cooldown & Hysteresis Protected)")
        col_breach, col_anom = st.columns(2)

        if alerts_df.empty:
            col_breach.success("No hazardous gas breach alerts triggered.")
            col_anom.success("No sensor hardware or distribution anomalies flagged.")
        else:
            if "category" not in alerts_df.columns:
                alerts_df["category"] = "GAS_BREACH"
            breach_df = alerts_df[alerts_df["category"] == "GAS_BREACH"].copy()
            anom_df = alerts_df[alerts_df["category"] == "SENSOR_ANOMALY"].copy()

            with col_breach:
                st.markdown(f"**Hazardous Gas Breach Forecasts (`{len(breach_df)}`)**")
                if breach_df.empty:
                    st.success("All gases within safe threshold bounds.")
                else:
                    breach_df["severity"] = breach_df["severity"].map(lambda s: SEVERITY_BADGE.get(s, s))
                    st.dataframe(
                        breach_df[["recorded_at", "severity", "gas", "message"]],
                        use_container_width=True,
                        hide_index=True,
                    )

            with col_anom:
                st.markdown(f"**Sensor Hardware & Anomaly Faults (`{len(anom_df)}`)**")
                if anom_df.empty:
                    st.success("No sensor anomalies detected by Isolation Forest.")
                else:
                    anom_df["severity"] = anom_df["severity"].map(lambda s: SEVERITY_BADGE.get(s, s))
                    st.dataframe(
                        anom_df[["recorded_at", "severity", "gas", "message"]],
                        use_container_width=True,
                        hide_index=True,
                    )

# ================= TAB 2: OFFLINE BENCHMARK =================
with tab_bench:
    st.subheader("Offline Multi-Horizon Evaluation on Unseen Test Split (1,067 Common Timestamps)")
    bench_rows = fetch_json("/benchmark", silent=True)
    if not bench_rows:
        local_bench = MODELS_DIR / "benchmark_report.json"
        if local_bench.exists():
            bench_rows = json.loads(local_bench.read_text())

    if not bench_rows:
        st.warning("Benchmark report not found. Run `python -m ml.evaluate` to generate `models/benchmark_report.json`.")
    else:
        bdf = pd.DataFrame(bench_rows)
        bdf["horizon_label"] = bdf["horizon_h"].astype(str) + "h"

        b1, b2, b3 = st.columns(3)
        for idx, (gas_key, gmeta) in enumerate(GAS_META.items()):
            gdf = bdf[bdf["gas"] == gas_key]
            fig_mae = px.bar(
                gdf,
                x="horizon_label",
                y="mae",
                color="model",
                barmode="group",
                title=f"{gmeta['label']} — Test MAE ({gmeta['unit']})",
                labels={"horizon_label": "Forecast Horizon", "mae": f"MAE ({gmeta['unit']})"},
            )
            fig_mae.update_layout(height=340)
            [b1, b2, b3][idx].plotly_chart(fig_mae, use_container_width=True)

        c_rec, c_lat = st.columns(2)
        with c_rec:
            fig_rec = px.bar(
                bdf,
                x="gas",
                y="breach_recall",
                color="model",
                facet_col="horizon_label",
                barmode="group",
                title="Hazardous Breach Recall by Model, Gas & Horizon (Higher = Safer)",
                labels={"breach_recall": "Breach Recall (0-1)", "gas": "Target Gas"},
            )
            fig_rec.update_layout(height=360)
            st.plotly_chart(fig_rec, use_container_width=True)

        with c_lat:
            lat_df = bdf.groupby("model", as_index=False)["latency_ms_per_packet"].first()
            fig_lat = px.bar(
                lat_df,
                x="model",
                y="latency_ms_per_packet",
                color="model",
                text="latency_ms_per_packet",
                title="Median Single-Packet Inference Latency (ms, Lower = Faster)",
                labels={"latency_ms_per_packet": "Latency (ms / packet)"},
            )
            fig_lat.update_layout(height=360)
            st.plotly_chart(fig_lat, use_container_width=True)

        st.markdown("**Full 27-Row Test Benchmark Table**")
        st.dataframe(bdf.drop(columns=["horizon_label"]), use_container_width=True, hide_index=True)

        # Optional: Neural network training/validation loss curves
        lstm_hist_path = MODELS_DIR / "lstm_history.json"
        cnn_hist_path = MODELS_DIR / "cnn_lstm_history.json"
        if lstm_hist_path.exists() and cnn_hist_path.exists():
            st.subheader("Neural Network Masked Huber Loss Curves (Early Stopping)")
            h_col1, h_col2 = st.columns(2)
            for col_ui, title, path in [
                (h_col1, "PyTorch Stacked LSTM (2-Layer, 54.9K params)", lstm_hist_path),
                (h_col2, "1D-CNN + LSTM Hybrid (Conv1d + LSTM, 30.3K params)", cnn_hist_path),
            ]:
                hdf = pd.DataFrame(json.loads(path.read_text()))
                fig_h = px.line(
                    hdf,
                    x="epoch",
                    y=["train_loss", "val_loss"],
                    markers=True,
                    title=title,
                    labels={"value": "Masked Huber Loss", "epoch": "Epoch"},
                )
                fig_h.update_layout(height=320)
                col_ui.plotly_chart(fig_h, use_container_width=True)

if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
