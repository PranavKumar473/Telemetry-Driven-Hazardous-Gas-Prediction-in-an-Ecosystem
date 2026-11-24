"""Sliding-window tensor builder for sequence models (MLP / 1D-CNN / LSTM / GRU).

Output: data/processed/windows.npz with, per split:
  X_* (N, SEQ_LEN, F)  scaled inputs, oldest -> newest timestep
  y_* (N, 9)           scaled targets (0.0 where invalid)
  m_* (N, 9)           target mask: 1 = real target, 0 = target hour was missing
  ts_* (N,)            timestamp of each window's last row
Scalers are fit on TRAIN ONLY and saved for inference.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ml.config import MODELS_DIR, RAW_PATH, SEQ_LEN, WINDOW_SCALERS_PATH, WINDOWS_PATH
from ml.features import HORIZONS, SENSOR_COLS, TARGET_GASES
from ml.preprocess import build_split_frames

# Per-timestep inputs: raw sensors + cyclical time. The sequence model learns lags itself.
WINDOW_FEATURES = SENSOR_COLS + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET_COLS = [f"{g}_target_{h}h" for g in TARGET_GASES for h in HORIZONS]


def fit_scalers(train_df: pd.DataFrame) -> tuple[StandardScaler, StandardScaler]:
    """Fit feature and target scalers on training rows only (NaNs are ignored by sklearn)."""
    x_scaler = StandardScaler().fit(train_df[WINDOW_FEATURES].to_numpy())
    y_scaler = StandardScaler().fit(train_df[TARGET_COLS].to_numpy())
    return x_scaler, y_scaler


def make_windows(
    X: np.ndarray, y: np.ndarray, timestamps: pd.Series, seq_len: int = SEQ_LEN
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Window k uses rows [i-seq_len+1 .. i] as input and the targets of row i.
    A window is kept only if:
      - its rows are seq_len consecutive hours,
      - every input value in it is finite (no long-gap NaNs),
      - at least one of its targets is real.

    Shapes: X (n, F), y (n, 9) -> X_w (N, T, F), y_w (N, 9), mask (N, 9), end_ts (N,)
    """
    hours = ((timestamps - timestamps.iloc[0]) / pd.Timedelta(hours=1)).to_numpy().astype(np.int64)
    ends = np.arange(seq_len - 1, len(X))
    offsets = np.arange(seq_len - 1, -1, -1)             # [T-1, ..., 1, 0]
    idx = ends[:, None] - offsets[None, :]                # (n_candidates, T), oldest -> newest

    contiguous = hours[ends] - hours[idx[:, 0]] == seq_len - 1
    inputs_ok = np.isfinite(X[idx]).all(axis=(1, 2))
    has_target = np.isfinite(y[ends]).any(axis=1)
    keep = contiguous & inputs_ok & has_target
    ends, idx = ends[keep], idx[keep]

    y_w = y[ends]
    mask = np.isfinite(y_w)
    return (
        X[idx].astype(np.float32),
        np.nan_to_num(y_w, nan=0.0).astype(np.float32),
        mask.astype(np.float32),
        timestamps.to_numpy()[ends],
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _, splits = build_split_frames(pd.read_csv(RAW_PATH))

    x_scaler, y_scaler = fit_scalers(splits["train"])
    joblib.dump(
        {"x_scaler": x_scaler, "y_scaler": y_scaler,
         "feature_cols": WINDOW_FEATURES, "target_cols": TARGET_COLS, "seq_len": SEQ_LEN},
        WINDOW_SCALERS_PATH,
    )

    arrays: dict[str, np.ndarray] = {
        "feature_names": np.array(WINDOW_FEATURES),
        "target_names": np.array(TARGET_COLS),
    }
    print(f"Building sliding windows (SEQ_LEN={SEQ_LEN}, F={len(WINDOW_FEATURES)}, targets={len(TARGET_COLS)})...")
    for name, df in splits.items():
        X_s = x_scaler.transform(df[WINDOW_FEATURES].to_numpy())
        y_s = y_scaler.transform(df[TARGET_COLS].to_numpy())
        X_w, y_w, m_w, end_ts = make_windows(X_s, y_s, df["timestamp"])
        arrays.update({f"X_{name}": X_w, f"y_{name}": y_w, f"m_{name}": m_w, f"ts_{name}": end_ts})
        print(
            f"  {name:<5} X={X_w.shape}  y={y_w.shape}  "
            f"valid targets={m_w.mean():.1%}  [{end_ts.min()} -> {end_ts.max()}]"
        )

    np.savez_compressed(WINDOWS_PATH, **arrays)
    print(f"Saved {WINDOWS_PATH.name} and {WINDOW_SCALERS_PATH.name}")


if __name__ == "__main__":
    main()
