"""End-to-end backend tests (SQLite, startup loading, sliding-window inference, alert cooldown, Pydantic v2)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import backend.db as db_module
from backend.alert_engine import DEFAULT_ALERT_MANAGER
from backend.main import app
from ml.config import PROCESSED_DIR


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run FastAPI TestClient against an isolated temporary SQLite database."""
    test_db = tmp_path / "test_telemetry.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    DEFAULT_ALERT_MANAGER.reset()
    with TestClient(app) as c:
        yield c
    DEFAULT_ALERT_MANAGER.reset()


def _sample_packets(n: int = 25) -> list[dict]:
    df = pd.read_csv(PROCESSED_DIR / "test_stream.csv").head(n)
    packets = []
    for _, row in df.iterrows():
        pkt = row.to_dict()
        pkt["node_id"] = "test_node_01"
        pkt["fault_mode"] = "normal"
        packets.append(pkt)
    return packets


def test_health_and_benchmark_endpoints(client: TestClient):
    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["status"] == "ok" and body["models_loaded"] is True
    assert set(body["loaded_models"]) == {"XGBoost", "IsolationForest", "PyTorch_LSTM", "CNN_LSTM"}
    assert body["seq_len"] == 24

    b = client.get("/benchmark")
    assert b.status_code == 200
    rows = b.json()
    assert len(rows) == 27
    assert {r["model"] for r in rows} == {"XGBoost", "PyTorch_LSTM", "CNN_LSTM"}


def test_ingest_warmup_and_all_three_models_predict(client: TestClient):
    packets = _sample_packets(25)

    # Packets 1..6: warming up (< 7 rows for 6h lag features)
    for i in range(6):
        resp = client.post("/ingest", json=packets[i])
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "WARMING_UP"
        assert data["xgboost"] is None and data["lstm"] is None and data["cnn_lstm"] is None

    # Packet 7 (index 6): XGBoost ready, neural nets still warming up (< 24 rows)
    resp7 = client.post("/ingest", json=packets[6])
    assert resp7.status_code == 200
    d7 = resp7.json()
    assert d7["status"] in {"SAFE", "ELEVATED", "WARNING", "CRITICAL"}
    assert d7["xgboost"] is not None
    assert d7["lstm"] is None and d7["cnn_lstm"] is None

    # Stream through packet 24 (index 23): all 3 models (XGBoost, GasLSTM, CNNLSTM) must produce finite physical units
    for i in range(7, 23):
        client.post("/ingest", json=packets[i])

    resp24 = client.post("/ingest", json=packets[23])
    assert resp24.status_code == 200
    d24 = resp24.json()
    for model_key in ("xgboost", "lstm", "cnn_lstm"):
        forecast = d24[model_key]
        assert forecast is not None, f"{model_key} should be active at 24 packets"
        vals = [forecast[k] for k in forecast if k != "latency_ms"]
        assert len(vals) == 9 and np.isfinite(vals).all()
        # Sanity check physical units: CO in mg/m^3, NO2 in ug/m^3
        assert -5.0 < forecast["co_pred_1h"] < 50.0
        assert 0.0 < forecast["no2_pred_1h"] < 600.0


def test_alert_cooldown_suppression_and_escalation(client: TestClient):
    packets = _sample_packets(12)
    # Warm up tabular window with 6 safe baseline packets
    for i in range(6):
        pkt = dict(packets[i], co_gt=1.0, no2_gt=50.0, c6h6_gt=3.0)
        client.post("/ingest", json=pkt)

    # Packet 7: WARNING spike on CO (co_gt=4.5 >= warning 4.0, < critical 6.0)
    warn_pkt = dict(packets[6], co_gt=4.5, no2_gt=50.0, c6h6_gt=3.0, fault_mode="leak")
    r_warn1 = client.post("/ingest", json=warn_pkt).json()
    co_alerts_1 = [a for a in r_warn1["alerts"] if a["gas"] == "co_gt" and a["category"] == "GAS_BREACH"]
    assert len(co_alerts_1) == 1 and co_alerts_1[0]["severity"] == "WARNING"

    # Packet 8: Repeated identical WARNING spike -> MUST be suppressed by cooldown
    warn_pkt2 = dict(packets[7], co_gt=4.6, no2_gt=50.0, c6h6_gt=3.0, fault_mode="leak")
    r_warn2 = client.post("/ingest", json=warn_pkt2).json()
    co_alerts_2 = [a for a in r_warn2["alerts"] if a["gas"] == "co_gt" and a["category"] == "GAS_BREACH"]
    assert len(co_alerts_2) == 0, "Duplicate WARNING alert should be suppressed during cooldown"
    assert r_warn2["status"] in {"WARNING", "CRITICAL"}

    # Packet 9: Escalation to CRITICAL (co_gt=8.5 >= critical 6.0) -> MUST bypass cooldown immediately
    crit_pkt = dict(packets[8], co_gt=8.5, no2_gt=50.0, c6h6_gt=3.0, fault_mode="leak")
    r_crit = client.post("/ingest", json=crit_pkt).json()
    co_alerts_3 = [a for a in r_crit["alerts"] if a["gas"] == "co_gt" and a["category"] == "GAS_BREACH"]
    assert len(co_alerts_3) == 1 and co_alerts_3[0]["severity"] == "CRITICAL"
    assert r_crit["status"] == "CRITICAL"

    # Packet 10: Repeated CRITICAL spike -> suppressed by cooldown
    crit_pkt2 = dict(packets[9], co_gt=8.6, no2_gt=50.0, c6h6_gt=3.0, fault_mode="leak")
    r_crit2 = client.post("/ingest", json=crit_pkt2).json()
    co_alerts_4 = [a for a in r_crit2["alerts"] if a["gas"] == "co_gt" and a["category"] == "GAS_BREACH"]
    assert len(co_alerts_4) == 0, "Duplicate CRITICAL alert should be suppressed during cooldown"


def test_history_and_reset_and_pydantic_validation(client: TestClient):
    packets = _sample_packets(8)
    for pkt in packets:
        client.post("/ingest", json=pkt)

    hist = client.get("/history", params={"node_id": "test_node_01", "limit": 50}).json()
    assert len(hist["telemetry"]) == 8
    assert len(hist["predictions"]) == 2  # packets 7 and 8 produced XGBoost predictions

    # Reset database and verify empty
    r_reset = client.post("/reset")
    assert r_reset.status_code == 200 and r_reset.json()["status"] == "reset_complete"
    hist_after = client.get("/history", params={"node_id": "test_node_01", "limit": 50}).json()
    assert hist_after == {"telemetry": [], "predictions": [], "alerts": []}

    # Strict Pydantic v2 validation: unknown field rejected with 422
    bad_pkt = dict(packets[0], unexpected_field=123)
    assert client.post("/ingest", json=bad_pkt).status_code == 422
