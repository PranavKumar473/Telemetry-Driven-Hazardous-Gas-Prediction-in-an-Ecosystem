"""Unified offline benchmark: XGBoost vs LSTM vs 1D-CNN+LSTM on the SAME test timestamps, in physical units."""
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor
from ml.config import MODELS_DIR, PROCESSED_DIR, WINDOW_SCALERS_PATH, WINDOWS_PATH
from ml.features import HORIZONS, TARGET_GASES
from ml.models import load_checkpoint
from ml.windowing import TARGET_COLS

BENCHMARK_PATH = MODELS_DIR / "benchmark_report.json"

# Hazardous WARNING thresholds used for breach recall (same as backend alert engine).
HAZARD_THRESHOLDS = {"co_gt": 4.0, "no2_gt": 150.0, "c6h6_gt": 15.0}

NN_MODELS = {"PyTorch_LSTM": "lstm_model.pt", "CNN_LSTM": "cnn_lstm_model.pt"}
LATENCY_WARMUP, LATENCY_RUNS = 10, 200


def breach_recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple[float | None, int]:
    """Share of real hazardous hours the model also predicted as hazardous. None if no breaches exist."""
    actual = y_true >= threshold
    n = int(actual.sum())
    if n == 0:
        return None, 0
    return float(((y_pred >= threshold) & actual).sum() / n), n


def median_latency_ms(fn) -> float:
    """Median wall-clock time of one call (models already loaded; warm-up runs excluded)."""
    for _ in range(LATENCY_WARMUP):
        fn()
    times = []
    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def align_test_sets() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Keep only test timestamps that have BOTH a tabular row (XGBoost) and a 24h window (NNs)."""
    data = np.load(WINDOWS_PATH)
    win_ts = pd.to_datetime(data["ts_test"])
    tab = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    tab["timestamp"] = pd.to_datetime(tab["timestamp"])

    in_both = win_ts.isin(tab["timestamp"])
    win_idx = np.flatnonzero(in_both)
    tab_aligned = tab.set_index("timestamp").loc[win_ts[win_idx]].reset_index()
    return tab_aligned, data["X_test"][win_idx], data["m_test"][win_idx].astype(bool)


def main() -> None:
    tab, X_win, mask = align_test_sets()
    y_true = tab[TARGET_COLS].to_numpy()
    print(f"Common test set: {len(tab):,} timestamps ({tab['timestamp'].min()} -> {tab['timestamp'].max()})")

    preds: dict[str, np.ndarray] = {}
    latency: dict[str, float] = {}

    # --- XGBoost: 9 regressors on tabular features (raw physical units) ---
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)
    X_tab = tab[feature_cols]
    xgb = {}
    for col in TARGET_COLS:
        gas, h = col.split("_target_")
        xgb[col] = XGBRegressor()
        xgb[col].load_model(MODELS_DIR / f"xgb_{gas}_{h}.json")
    preds["XGBoost"] = np.column_stack([xgb[c].predict(X_tab) for c in TARGET_COLS])
    one_row = X_tab.iloc[[0]]
    latency["XGBoost"] = median_latency_ms(lambda: [m.predict(one_row) for m in xgb.values()])

    # --- Neural nets: scaled outputs -> physical units via the train-fit y_scaler ---
    y_scaler = joblib.load(WINDOW_SCALERS_PATH)["y_scaler"]
    X_t = torch.from_numpy(X_win)
    for name, fname in NN_MODELS.items():
        model, _ = load_checkpoint(MODELS_DIR / fname)
        with torch.no_grad():
            preds[name] = y_scaler.inverse_transform(model(X_t).numpy())
        one_win = X_t[:1]

        def single(model=model, one_win=one_win):
            with torch.no_grad():
                return y_scaler.inverse_transform(model(one_win).numpy())

        latency[name] = median_latency_ms(single)

    # --- Metrics per (model, gas, horizon) on real (unmasked) targets only ---
    rows = []
    for model_name, y_hat in preds.items():
        for j, col in enumerate(TARGET_COLS):
            gas, h = col.split("_target_")[0], int(col.split("_")[-1][:-1])
            valid = mask[:, j]
            yt, yp = y_true[valid, j], y_hat[valid, j]
            recall, n_breaches = breach_recall(yt, yp, HAZARD_THRESHOLDS[gas])
            rows.append({
                "model": model_name,
                "gas": gas,
                "horizon_h": h,
                "mae": round(float(mean_absolute_error(yt, yp)), 3),
                "rmse": round(float(root_mean_squared_error(yt, yp)), 3),
                "breach_recall": None if recall is None else round(recall, 3),
                "n_breaches": n_breaches,
                "n_samples": int(valid.sum()),
                "latency_ms_per_packet": round(latency[model_name], 4),
            })

    report = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print("\n=== OFFLINE BENCHMARK (unseen test split, physical units) ===")
    print(report.to_string(index=False))
    print("\n=== Summary (mean over 3 gases x 3 horizons; MAE/RMSE not unit-normalised) ===")
    print(report.groupby("model", sort=False)[["breach_recall", "latency_ms_per_packet"]].mean().round(3).to_string())
    print("\n=== MAE by gas (mean over horizons) ===")
    print(report.pivot_table(index="gas", columns="model", values="mae", aggfunc="mean", sort=False).round(3).to_string())

    with open(BENCHMARK_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} rows to {BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
