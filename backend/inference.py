"""Live sliding-window inference engine for XGBoost, Isolation Forest, GasLSTM, and CNNLSTM."""
import json
import time
from typing import Any
import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
from ml.config import MODELS_DIR, SEQ_LEN, WINDOW_SCALERS_PATH
from ml.features import HORIZONS, TARGET_GASES, build_features
from ml.models import load_checkpoint
from ml.windowing import TARGET_COLS, WINDOW_FEATURES

MIN_TABULAR_ROWS = max(HORIZONS) + 1  # 7 rows required so 6h lag/diff features are non-null


def _format_target_map(pred_map: dict[str, float]) -> dict[str, float]:
    return {
        "co_pred_1h": float(pred_map["co_gt_target_1h"]),
        "co_pred_3h": float(pred_map["co_gt_target_3h"]),
        "co_pred_6h": float(pred_map["co_gt_target_6h"]),
        "no2_pred_1h": float(pred_map["no2_gt_target_1h"]),
        "no2_pred_3h": float(pred_map["no2_gt_target_3h"]),
        "no2_pred_6h": float(pred_map["no2_gt_target_6h"]),
        "c6h6_pred_1h": float(pred_map["c6h6_gt_target_1h"]),
        "c6h6_pred_3h": float(pred_map["c6h6_gt_target_3h"]),
        "c6h6_pred_6h": float(pred_map["c6h6_gt_target_6h"]),
    }


class LiveInferenceEngine:
    """Loads all trained models and scalers once at startup and serves sliding-window forecasts."""

    def __init__(self) -> None:
        with open(MODELS_DIR / "feature_cols.json") as f:
            self.feature_cols: list[str] = json.load(f)

        # 1. Load 9 multi-horizon XGBoost regressors (39 tabular features)
        self.xgb_models: dict[str, XGBRegressor] = {}
        for gas in TARGET_GASES:
            for h in HORIZONS:
                key = f"{gas}_target_{h}h"
                model = XGBRegressor()
                model.load_model(MODELS_DIR / f"xgb_{gas}_{h}h.json")
                self.xgb_models[key] = model

        # 2. Load Isolation Forest anomaly detector (39 tabular features)
        self.iso_forest = joblib.load(MODELS_DIR / "isolation_forest.joblib")

        # 3. Load sequence scalers (16 WINDOW_FEATURES -> 9 TARGET_COLS)
        scaler_bundle = joblib.load(WINDOW_SCALERS_PATH)
        self.x_scaler = scaler_bundle["x_scaler"]
        self.y_scaler = scaler_bundle["y_scaler"]
        self.window_features: list[str] = list(scaler_bundle.get("feature_cols", WINDOW_FEATURES))
        self.target_cols: list[str] = list(scaler_bundle.get("target_cols", TARGET_COLS))
        self.seq_len: int = int(scaler_bundle.get("seq_len", SEQ_LEN))

        # 4. Load PyTorch GasLSTM & 1D-CNN+LSTM checkpoints once in eval() mode
        self.lstm, _ = load_checkpoint(MODELS_DIR / "lstm_model.pt")
        self.cnn_lstm, _ = load_checkpoint(MODELS_DIR / "cnn_lstm_model.pt")

    def _predict_nn(self, model: torch.nn.Module, x_tensor: torch.Tensor) -> dict[str, float]:
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_s = model(x_tensor).numpy()
        pred_phys = self.y_scaler.inverse_transform(pred_s)[0]
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        pred_map = {col: float(pred_phys[i]) for i, col in enumerate(self.target_cols)}
        return {**_format_target_map(pred_map), "latency_ms": latency_ms}

    def predict(self, window_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Run XGBoost + Isolation Forest once >= 7 rows exist, and run
        GasLSTM + CNNLSTM once >= 24 rows exist in the sliding window.
        """
        n_rows = len(window_rows)
        if n_rows < MIN_TABULAR_ROWS:
            return {
                "ready": False,
                "reason": (
                    f"Warming up sliding window ({n_rows}/{MIN_TABULAR_ROWS} packets for XGBoost, "
                    f"{n_rows}/{self.seq_len} for neural nets)"
                ),
            }

        df = pd.DataFrame(window_rows)
        feat_df = build_features(df)
        latest_tab = feat_df[self.feature_cols].iloc[[-1]]
        if latest_tab.isna().any(axis=None):
            return {
                "ready": False,
                "reason": f"Warming up lag features ({n_rows}/{MIN_TABULAR_ROWS} packets)",
            }

        # --- 1. Tabular XGBoost (39 features) + Isolation Forest ---
        t0 = time.perf_counter()
        xgb_map = {col: float(model.predict(latest_tab)[0]) for col, model in self.xgb_models.items()}
        is_anomaly = bool(self.iso_forest.predict(latest_tab)[0] == -1)
        xgb_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        xgb_out = {**_format_target_map(xgb_map), "latency_ms": xgb_ms}

        # --- 2. Sequence Neural Networks: (1, 24, 16) tensor -> physical units ---
        lstm_out = None
        cnn_lstm_out = None
        if len(feat_df) >= self.seq_len:
            seq_raw = feat_df[self.window_features].iloc[-self.seq_len :].to_numpy(dtype=np.float32)
            if np.isfinite(seq_raw).all():
                seq_scaled = self.x_scaler.transform(seq_raw).astype(np.float32)
                x_tensor = torch.from_numpy(seq_scaled).unsqueeze(0)  # shape: (1, 24, 16)
                lstm_out = self._predict_nn(self.lstm, x_tensor)
                cnn_lstm_out = self._predict_nn(self.cnn_lstm, x_tensor)

        return {
            "ready": True,
            "is_anomaly": is_anomaly,
            "xgboost": xgb_out,
            "lstm": lstm_out,
            "cnn_lstm": cnn_lstm_out,
        }
