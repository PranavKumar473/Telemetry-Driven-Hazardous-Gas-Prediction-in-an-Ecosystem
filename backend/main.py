import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from backend.db import (
    init_db,
    insert_telemetry,
    fetch_recent_window,
    insert_prediction,
    insert_alerts,
    fetch_history,
    reset_db,
)
from backend.inference import LiveInferenceEngine
from backend.alert_engine import evaluate_alerts

app = FastAPI(title="Telemetry-Driven Hazardous Gas Prediction API")
engine: LiveInferenceEngine | None = None


class TelemetryPacket(BaseModel):
    node_id: str = "eco_node_01"
    timestamp: str
    co_gt: float
    pt08_s1_co: float
    c6h6_gt: float
    pt08_s2_nmhc: float
    nox_gt: float
    pt08_s3_nox: float
    no2_gt: float
    pt08_s4_no2: float
    pt08_s5_o3: float
    temp: float
    rh: float
    ah: float
    fault_mode: str = "normal"


@app.on_event("startup")
def startup() -> None:
    global engine
    init_db()
    engine = LiveInferenceEngine()


@app.post("/ingest")
def ingest_packet(packet: TelemetryPacket):
    payload = packet.model_dump()
    insert_telemetry(payload)

    window = fetch_recent_window(packet.node_id, limit=24)
    result = engine.predict(window)

    if not result["ready"]:
        return {"status": "WARMING_UP", "reason": result["reason"]}

    xgb = result["xgboost"]
    insert_prediction({
        "node_id": packet.node_id,
        "recorded_at": packet.timestamp,
        "model_name": "XGBoost",
        "is_anomaly": result["is_anomaly"],
        **xgb,
    })

    if result["lstm"]:
        insert_prediction({
            "node_id": packet.node_id,
            "recorded_at": packet.timestamp,
            "model_name": "PyTorch_LSTM",
            "is_anomaly": result["is_anomaly"],
            **result["lstm"],
        })

    overall_status, alerts = evaluate_alerts(payload, xgb, result["is_anomaly"])
    insert_alerts(packet.node_id, packet.timestamp, alerts)

    return {
        "status": overall_status,
        "is_anomaly": result["is_anomaly"],
        "xgboost": xgb,
        "lstm": result["lstm"],
        "alerts": alerts,
    }


@app.get("/history")
def get_history(node_id: str = "eco_node_01", limit: int = 120):
    return fetch_history(node_id=node_id, limit=limit)


@app.get("/benchmark")
def get_benchmark():
    path = Path("models/benchmark_report.json")
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


@app.post("/reset")
def reset_stream():
    reset_db()
    return {"status": "reset_complete"}