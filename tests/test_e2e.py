"""End-to-end integration test streaming 30+ simulator packets (normal & leak modes) into FastAPI."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import backend.db as db_module
from backend.alert_engine import DEFAULT_ALERT_MANAGER
from backend.main import app
from simulator.producer import STREAM_PATH, build_packet


@pytest.fixture()
def e2e_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    test_db = tmp_path / "e2e_telemetry.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    DEFAULT_ALERT_MANAGER.reset()
    with TestClient(app) as client:
        yield client
    DEFAULT_ALERT_MANAGER.reset()


def test_e2e_normal_stream_30_packets_activates_all_three_models(e2e_client: TestClient):
    df = pd.read_csv(STREAM_PATH).head(30)
    assert len(df) == 30

    statuses = []
    for step, (_, row) in enumerate(df.iterrows(), start=1):
        pkt = build_packet(row.to_dict(), node_id="eco_node_e2e", step=step, mode="normal")
        assert pkt is not None
        resp = e2e_client.post("/ingest", json=pkt)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        statuses.append(body["status"])

        if step <= 6:
            assert body["status"] == "WARMING_UP"
            assert body["xgboost"] is None and body["lstm"] is None and body["cnn_lstm"] is None
        elif step <= 23:
            assert body["status"] != "WARMING_UP"
            assert body["xgboost"] is not None
            assert body["lstm"] is None and body["cnn_lstm"] is None
        else:
            # Steps 24..30: all 3 models active with finite physical predictions
            for key in ("xgboost", "lstm", "cnn_lstm"):
                assert body[key] is not None, f"{key} missing at step {step}"
                preds = [v for k, v in body[key].items() if k != "latency_ms"]
                assert len(preds) == 9 and np.isfinite(preds).all()

    history = e2e_client.get("/history", params={"node_id": "eco_node_e2e", "limit": 100}).json()
    assert len(history["telemetry"]) == 30
    # 24 steps with XGBoost (steps 7..30) + 7 steps with LSTM (steps 24..30) + 7 steps with CNN_LSTM (steps 24..30) = 38
    assert len(history["predictions"]) == 24 + 7 + 7
    assert {p["model_name"] for p in history["predictions"]} == {"XGBoost", "PyTorch_LSTM", "CNN_LSTM"}


def test_e2e_leak_and_packet_loss_modes(e2e_client: TestClient):
    df = pd.read_csv(STREAM_PATH).head(32)
    breach_alerts_emitted = 0
    critical_steps = 0

    for step, (_, row) in enumerate(df.iterrows(), start=1):
        pkt = build_packet(row.to_dict(), node_id="eco_node_leak", step=step, mode="leak")
        assert pkt is not None
        resp = e2e_client.post("/ingest", json=pkt)
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] == "CRITICAL":
            critical_steps += 1
        breach_alerts_emitted += sum(1 for a in body["alerts"] if a["category"] == "GAS_BREACH")

    # Leak ramp starts at step 20 and must trigger gas breach alerts and critical status,
    # while cooldown prevents emitting 3 gas alerts on every single post-20 step (< 20 total).
    assert critical_steps >= 1
    assert 1 <= breach_alerts_emitted < 20

    # Verify packet_loss mode with seeded RNG drops ~15% of packets cleanly and remaining packets ingest fine
    rng = random.Random(42)
    dropped = 0
    sent = 0
    for step, (_, row) in enumerate(df.iterrows(), start=1):
        pkt = build_packet(row.to_dict(), node_id="eco_node_loss", step=step, mode="packet_loss", rng=rng)
        if pkt is None:
            dropped += 1
        else:
            sent += 1
            assert e2e_client.post("/ingest", json=pkt).status_code == 200
    assert dropped >= 1 and sent + dropped == len(df)
