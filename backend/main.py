"""FastAPI ingestion, sliding-window multi-model inference, alerting, and history service."""
from contextlib import asynccontextmanager
import json
import math
from typing import Literal
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from backend.alert_engine import DEFAULT_ALERT_MANAGER, evaluate_alerts
from backend.db import (
    fetch_history,
    fetch_recent_window,
    init_db,
    insert_alerts,
    insert_prediction,
    insert_telemetry,
    reset_db,
)
from backend.inference import LiveInferenceEngine
from ml.config import MODELS_DIR, SEQ_LEN

engine: LiveInferenceEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine
    init_db()
    engine = LiveInferenceEngine()
    yield


app = FastAPI(
    title="Telemetry-Driven Hazardous Gas Prediction API",
    version="1.3.0",
    lifespan=lifespan,
)


def _get_engine() -> LiveInferenceEngine:
    global engine
    if engine is None:
        engine = LiveInferenceEngine()
    return engine


# ---------------- Pydantic v2 Request / Response Schemas ----------------

class TelemetryPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(default="eco_node_01", min_length=1, max_length=64)
    timestamp: str = Field(..., min_length=1)
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
    fault_mode: Literal["normal", "leak", "drift", "packet_loss"] = "normal"

    @field_validator(
        "co_gt", "pt08_s1_co", "c6h6_gt", "pt08_s2_nmhc", "nox_gt", "pt08_s3_nox",
        "no2_gt", "pt08_s4_no2", "pt08_s5_o3", "temp", "rh", "ah"
    )
    @classmethod
    def _must_be_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Sensor reading must be a finite float.")
        return float(v)


class ModelForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    co_pred_1h: float
    co_pred_3h: float
    co_pred_6h: float
    no2_pred_1h: float
    no2_pred_3h: float
    no2_pred_6h: float
    c6h6_pred_1h: float
    c6h6_pred_3h: float
    c6h6_pred_6h: float
    latency_ms: float


class AlertItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["ELEVATED", "WARNING", "CRITICAL"]
    category: Literal["GAS_BREACH", "SENSOR_ANOMALY"] = "GAS_BREACH"
    gas: str
    message: str


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["WARMING_UP", "SAFE", "ELEVATED", "WARNING", "CRITICAL"]
    reason: str | None = None
    is_anomaly: bool = False
    xgboost: ModelForecast | None = None
    lstm: ModelForecast | None = None
    cnn_lstm: ModelForecast | None = None
    alerts: list[AlertItem] = Field(default_factory=list)


class TelemetryRecord(BaseModel):
    id: int
    node_id: str
    recorded_at: str
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
    fault_mode: str


class PredictionRecord(BaseModel):
    id: int
    node_id: str
    recorded_at: str
    model_name: str
    co_pred_1h: float
    co_pred_3h: float
    co_pred_6h: float
    no2_pred_1h: float
    no2_pred_3h: float
    no2_pred_6h: float
    c6h6_pred_1h: float
    c6h6_pred_3h: float
    c6h6_pred_6h: float
    is_anomaly: int
    latency_ms: float


class AlertRecord(BaseModel):
    id: int
    node_id: str
    recorded_at: str
    severity: str
    category: str
    gas: str
    message: str


class HistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    telemetry: list[TelemetryRecord]
    predictions: list[PredictionRecord]
    alerts: list[AlertRecord]


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    gas: str
    horizon_h: int
    mae: float
    rmse: float
    breach_recall: float | None = None
    n_breaches: int = 0
    n_samples: int = 0
    latency_ms_per_packet: float


class ResetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["reset_complete"]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    models_loaded: bool
    loaded_models: list[str]
    seq_len: int


# ---------------- API Endpoints ----------------

@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    eng = _get_engine()
    return HealthResponse(
        status="ok",
        models_loaded=eng is not None,
        loaded_models=["XGBoost", "IsolationForest", "PyTorch_LSTM", "CNN_LSTM"],
        seq_len=eng.seq_len,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest_packet(packet: TelemetryPacket) -> IngestResponse:
    eng = _get_engine()
    payload = packet.model_dump()
    insert_telemetry(payload)

    window = fetch_recent_window(packet.node_id, limit=SEQ_LEN)
    result = eng.predict(window)

    if not result["ready"]:
        return IngestResponse(status="WARMING_UP", reason=result["reason"])

    for model_name, key in [
        ("XGBoost", "xgboost"),
        ("PyTorch_LSTM", "lstm"),
        ("CNN_LSTM", "cnn_lstm"),
    ]:
        if result.get(key) is not None:
            insert_prediction({
                "node_id": packet.node_id,
                "recorded_at": packet.timestamp,
                "model_name": model_name,
                "is_anomaly": result["is_anomaly"],
                **result[key],
            })

    overall_status, alerts = evaluate_alerts(
        actual=payload,
        xgb_pred=result["xgboost"],
        is_anomaly=result["is_anomaly"],
        lstm_pred=result["lstm"],
        cnn_lstm_pred=result["cnn_lstm"],
    )
    insert_alerts(packet.node_id, packet.timestamp, alerts)

    return IngestResponse(
        status=overall_status,
        is_anomaly=result["is_anomaly"],
        xgboost=ModelForecast(**result["xgboost"]),
        lstm=ModelForecast(**result["lstm"]) if result["lstm"] else None,
        cnn_lstm=ModelForecast(**result["cnn_lstm"]) if result["cnn_lstm"] else None,
        alerts=[AlertItem(**a) for a in alerts],
    )


@app.get("/history", response_model=HistoryResponse)
def get_history(
    node_id: str = Query(default="eco_node_01", min_length=1),
    limit: int = Query(default=120, ge=1, le=1000),
) -> HistoryResponse:
    data = fetch_history(node_id=node_id, limit=limit)
    return HistoryResponse(**data)


@app.get("/benchmark", response_model=list[BenchmarkItem])
def get_benchmark() -> list[BenchmarkItem]:
    path = MODELS_DIR / "benchmark_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Benchmark report not found. Run `python -m ml.evaluate`.")
    with open(path) as f:
        rows = json.load(f)
    return [BenchmarkItem(**r) for r in rows]


@app.post("/reset", response_model=ResetResponse)
def reset_stream() -> ResetResponse:
    reset_db()
    DEFAULT_ALERT_MANAGER.reset()
    return ResetResponse(status="reset_complete")
