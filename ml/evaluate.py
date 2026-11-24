import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor
from ml.features import TARGET_GASES, HORIZONS
from ml.train_lstm import GasLSTM, SEQ_LEN, TARGET_COLS, make_sequences

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

# Hazardous warning thresholds for breach recall evaluation
HAZARD_THRESHOLDS = {
    "co_gt": 4.0,     # mg/m^3
    "no2_gt": 150.0,  # ug/m^3
    "c6h6_gt": 15.0,  # ug/m^3
}


def breach_recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    actual_breaches = y_true >= threshold
    if actual_breaches.sum() == 0:
        return 1.0
    caught = (y_pred >= threshold) & actual_breaches
    return float(caught.sum() / actual_breaches.sum())


def main() -> None:
    test_df = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    with open(MODELS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    # Align test rows so XGBoost and LSTM are evaluated on identical timestamps
    aligned_test = test_df.iloc[SEQ_LEN - 1 :].reset_index(drop=True)
    X_test_tab = aligned_test[feature_cols]

    # --- 1. Evaluate XGBoost ---
    xgb_preds = {}
    t0 = time.perf_counter()
    for gas in TARGET_GASES:
        for h in HORIZONS:
            model = XGBRegressor()
            model.load_model(MODELS_DIR / f"xgb_{gas}_{h}h.json")
            xgb_preds[f"{gas}_target_{h}h"] = model.predict(X_test_tab)
    xgb_latency_ms = ((time.perf_counter() - t0) / len(aligned_test)) * 1000.0

    # --- 2. Evaluate PyTorch LSTM ---
    scalers = joblib.load(MODELS_DIR / "lstm_scalers.joblib")
    ckpt = torch.load(MODELS_DIR / "lstm_model.pt", map_location="cpu", weights_only=True)
    lstm = GasLSTM(input_size=ckpt["input_size"], output_size=ckpt["output_size"])
    lstm.load_state_dict(ckpt["state_dict"])
    lstm.eval()

    X_test_s = scalers["x_scaler"].transform(test_df[feature_cols].values)
    y_dummy = np.zeros((len(test_df), len(TARGET_COLS)), dtype=np.float32)
    X_test_seq, _ = make_sequences(X_test_s, y_dummy, seq_len=SEQ_LEN)

    t0 = time.perf_counter()
    with torch.no_grad():
        lstm_out_s = lstm(torch.from_numpy(X_test_seq)).numpy()
    lstm_out = scalers["y_scaler"].inverse_transform(lstm_out_s)
    lstm_latency_ms = ((time.perf_counter() - t0) / len(aligned_test)) * 1000.0

    lstm_preds = {col: lstm_out[:, idx] for idx, col in enumerate(TARGET_COLS)}

    # --- 3. Compile Benchmark Report ---
    rows = []
    for gas in TARGET_GASES:
        thresh = HAZARD_THRESHOLDS[gas]
        for h in HORIZONS:
            col = f"{gas}_target_{h}h"
            y_true = aligned_test[col].values

            for model_name, pred_dict, lat in [
                ("XGBoost", xgb_preds, xgb_latency_ms),
                ("PyTorch_LSTM", lstm_preds, lstm_latency_ms),
            ]:
                y_hat = pred_dict[col]
                rows.append({
                    "model": model_name,
                    "gas": gas,
                    "horizon_h": h,
                    "mae": round(float(mean_absolute_error(y_true, y_hat)), 3),
                    "rmse": round(float(root_mean_squared_error(y_true, y_hat)), 3),
                    "breach_recall": round(breach_recall(y_true, y_hat, thresh), 3),
                    "latency_ms_per_packet": round(lat, 4),
                })

    report_df = pd.DataFrame(rows)
    print("\n=== OFFLINE BENCHMARK: XGBoost vs PyTorch LSTM (Unseen Test Set) ===")
    print(report_df.to_string(index=False))

    with open(MODELS_DIR / "benchmark_report.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved benchmark report to: {(MODELS_DIR / 'benchmark_report.json').resolve()}")


if __name__ == "__main__":
    main()