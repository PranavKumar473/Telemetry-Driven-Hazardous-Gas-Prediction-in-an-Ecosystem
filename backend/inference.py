import json
import time
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
from ml.features import TARGET_GASES, HORIZONS, build_features
from ml.train_lstm import GasLSTM, SEQ_LEN, TARGET_COLS

MODELS_DIR = Path("models")


class LiveInferenceEngine:
    def __init__(self) -> None:
        with open(MODELS_DIR / "feature_cols.json") as f:
            self.feature_cols: list[str] = json.load(f)

        # Load XGBoost models
        self.xgb_models: dict[str, XGBRegressor] = {}
        for gas in TARGET_GASES:
            for h in HORIZONS:
                key = f"{gas}_{h}h"
                model = XGBRegressor()
                model.load_model(MODELS_DIR / f"xgb_{gas}_{h}h.json")
                self.xgb_models[key] = model

        # Load Isolation Forest
        self.iso_forest = joblib.load(MODELS_DIR / "isolation_forest.joblib")

        # Load PyTorch LSTM + scalers
        self.scalers = joblib.load(MODELS_DIR / "lstm_scalers.joblib")
        ckpt = torch.load(MODELS_DIR / "lstm_model.pt", map_location="cpu", weights_only=True)
        self.lstm = GasLSTM(input_size=ckpt["input_size"], output_size=ckpt["output_size"])
        self.lstm.load_state_dict(ckpt["state_dict"])
        self.lstm.eval()

    def predict(self, window_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Run XGBoost, Isolation Forest, and LSTM on the latest sliding window."""
        df = pd.DataFrame(window_rows)
        feat_df = build_features(df).dropna(subset=self.feature_cols)

        if feat_df.empty:
            return {"ready": False, "reason": "Warming up sliding window (need >= 7 packets)"}

        latest_X = feat_df[self.feature_cols].iloc[[-1]]

        # 1. XGBoost + Isolation Forest
        t0 = time.perf_counter()
        xgb_out = {
            "co_pred_1h": float(self.xgb_models["co_gt_1h"].predict(latest_X)[0]),
            "co_pred_3h": float(self.xgb_models["co_gt_3h"].predict(latest_X)[0]),
            "co_pred_6h": float(self.xgb_models["co_gt_6h"].predict(latest_X)[0]),
            "no2_pred_1h": float(self.xgb_models["no2_gt_1h"].predict(latest_X)[0]),
            "no2_pred_3h": float(self.xgb_models["no2_gt_3h"].predict(latest_X)[0]),
            "no2_pred_6h": float(self.xgb_models["no2_gt_6h"].predict(latest_X)[0]),
            "c6h6_pred_1h": float(self.xgb_models["c6h6_gt_1h"].predict(latest_X)[0]),
            "c6h6_pred_3h": float(self.xgb_models["c6h6_gt_3h"].predict(latest_X)[0]),
            "c6h6_pred_6h": float(self.xgb_models["c6h6_gt_6h"].predict(latest_X)[0]),
        }
        is_anomaly = bool(self.iso_forest.predict(latest_X)[0] == -1)
        xgb_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        # 2. PyTorch LSTM (runs once >= SEQ_LEN valid feature rows exist)
        lstm_out = None
        lstm_ms = None
        if len(feat_df) >= SEQ_LEN:
            t1 = time.perf_counter()
            seq_raw = feat_df[self.feature_cols].iloc[-SEQ_LEN:].values
            seq_s = self.scalers["x_scaler"].transform(seq_raw).astype(np.float32)
            with torch.no_grad():
                pred_s = self.lstm(torch.from_numpy(seq_s).unsqueeze(0)).numpy()
            pred = self.scalers["y_scaler"].inverse_transform(pred_s)[0]
            pred_map = {col: float(pred[i]) for i, col in enumerate(TARGET_COLS)}
            lstm_out = {
                "co_pred_1h": pred_map["co_gt_target_1h"],
                "co_pred_3h": pred_map["co_gt_target_3h"],
                "co_pred_6h": pred_map["co_gt_target_6h"],
                "no2_pred_1h": pred_map["no2_gt_target_1h"],
                "no2_pred_3h": pred_map["no2_gt_target_3h"],
                "no2_pred_6h": pred_map["no2_gt_target_6h"],
                "c6h6_pred_1h": pred_map["c6h6_gt_target_1h"],
                "c6h6_pred_3h": pred_map["c6h6_gt_target_3h"],
                "c6h6_pred_6h": pred_map["c6h6_gt_target_6h"],
            }
            lstm_ms = round((time.perf_counter() - t1) * 1000.0, 3)

        return {
            "ready": True,
            "is_anomaly": is_anomaly,
            "xgboost": {**xgb_out, "latency_ms": xgb_ms},
            "lstm": ({**lstm_out, "latency_ms": lstm_ms} if lstm_out else None),
        }