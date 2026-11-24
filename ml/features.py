import numpy as np
import pandas as pd

TARGET_GASES = ["co_gt", "no2_gt", "c6h6_gt"]
SENSOR_COLS = [
    "co_gt", "pt08_s1_co", "c6h6_gt", "pt08_s2_nmhc",
    "nox_gt", "pt08_s3_nox", "no2_gt", "pt08_s4_no2",
    "pt08_s5_o3", "temp", "rh", "ah",
]
HORIZONS = [1, 3, 6]  # Predict 1h, 3h, and 6h ahead


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series features from cleaned telemetry data.
    Expects a chronologically sorted DataFrame with a 'timestamp' column.
    """
    feat = df.copy()
    ts = pd.to_datetime(feat["timestamp"])

    # 1. Cyclical time features (captures daily traffic/temperature cycles)
    feat["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    feat["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)

    # 2. Lag, rolling, and rate-of-change features for target gases
    for col in TARGET_GASES:
        for lag in [1, 3, 6]:
            feat[f"{col}_lag_{lag}h"] = feat[col].shift(lag)

        feat[f"{col}_roll_mean_6h"] = feat[col].rolling(window=6, min_periods=1).mean()
        feat[f"{col}_roll_std_6h"] = feat[col].rolling(window=6, min_periods=1).std().fillna(0.0)

        # Rate of change (derivative) to detect rapid leak buildup
        feat[f"{col}_diff_1h"] = feat[col].diff(1)
        feat[f"{col}_diff_3h"] = feat[col].diff(3)

    # 3. Short-term weather shifts (humidity/temp inversions trap hazardous gases)
    feat["temp_diff_1h"] = feat["temp"].diff(1)
    feat["rh_diff_1h"] = feat["rh"].diff(1)

    return feat


def add_multi_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add shifted future target columns for t+1h, t+3h, and t+6h forecasting."""
    out = df.copy()
    for col in TARGET_GASES:
        for h in HORIZONS:
            out[f"{col}_target_{h}h"] = out[col].shift(-h)
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all model input feature columns (excludes timestamp and targets)."""
    exclude = {"timestamp"} | {
        f"{col}_target_{h}h" for col in TARGET_GASES for h in HORIZONS
    }
    return [c for c in df.columns if c not in exclude]