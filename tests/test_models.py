"""Model shape / finiteness checks and masked-loss behaviour for Sub-Phase 1.2."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from xgboost import XGBRegressor
from ml.config import MODELS_DIR, PROCESSED_DIR, SEQ_LEN, WINDOWS_PATH
from ml.models import CNNLSTM, GasLSTM, load_checkpoint
from ml.nn_training import masked_huber_loss
from ml.windowing import TARGET_COLS, WINDOW_FEATURES

F, T, K = len(WINDOW_FEATURES), SEQ_LEN, len(TARGET_COLS)


# ---------------- Unit tests ----------------

@pytest.mark.parametrize("model_cls", [GasLSTM, CNNLSTM])
def test_forward_shape_and_finite(model_cls):
    torch.manual_seed(0)
    model = model_cls(input_size=F, output_size=K).eval()
    with torch.no_grad():
        out = model(torch.randn(4, T, F))
    assert out.shape == (4, K)
    assert torch.isfinite(out).all()


def test_masked_loss_ignores_masked_targets():
    torch.manual_seed(0)
    pred = torch.randn(5, K, requires_grad=True)
    target = torch.randn(5, K)
    mask = torch.ones(5, K)
    mask[0, :3] = 0.0
    loss_a = masked_huber_loss(pred, target, mask)

    corrupted = target.clone()
    corrupted[0, :3] = 1e6                              # garbage where mask == 0
    loss_b = masked_huber_loss(pred, corrupted, mask)
    assert torch.isclose(loss_a, loss_b)

    loss_b.backward()
    assert (pred.grad[0, :3] == 0).all()                # no learning signal from missing targets
    assert (pred.grad[mask.bool()] != 0).any()


# ---------------- Artifact tests (trained models) ----------------

trained = pytest.mark.skipif(
    not (MODELS_DIR / "cnn_lstm_model.pt").exists(), reason="Run the ml.train_* scripts first."
)


@trained
@pytest.mark.parametrize("fname", ["lstm_model.pt", "cnn_lstm_model.pt"])
def test_checkpoints_load_and_predict(fname):
    model, ckpt = load_checkpoint(MODELS_DIR / fname)
    assert ckpt["feature_cols"] == WINDOW_FEATURES and ckpt["target_cols"] == TARGET_COLS
    X = torch.from_numpy(np.load(WINDOWS_PATH)["X_test"][:8])
    with torch.no_grad():
        out = model(X)
    assert out.shape == (8, K) and torch.isfinite(out).all()


@trained
def test_xgboost_and_isolation_forest_predict_finite():
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    X = pd.read_csv(PROCESSED_DIR / "test_features.csv")[feature_cols].head(8)
    for col in TARGET_COLS:
        gas, h = col.split("_target_")
        model = XGBRegressor()
        model.load_model(MODELS_DIR / f"xgb_{gas}_{h}.json")
        pred = model.predict(X)
        assert pred.shape == (8,) and np.isfinite(pred).all()
    iso = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    assert set(iso.predict(X)) <= {-1, 1}


@trained
def test_benchmark_report_schema():
    path = MODELS_DIR / "benchmark_report.json"
    if not path.exists():
        pytest.skip("Run ml.evaluate first.")
    rows = json.loads(path.read_text())
    keys = {"model", "gas", "horizon_h", "mae", "rmse", "breach_recall", "n_breaches", "n_samples", "latency_ms_per_packet"}
    assert len(rows) == 27
    assert {r["model"] for r in rows} == {"XGBoost", "PyTorch_LSTM", "CNN_LSTM"}
    for r in rows:
        assert keys <= r.keys()
        assert np.isfinite([r["mae"], r["rmse"], r["latency_ms_per_packet"]]).all()
