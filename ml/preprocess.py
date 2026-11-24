import numpy as np
import pandas as pd
from ml.config import (
    MAX_FFILL_HOURS,
    PROCESSED_DIR,
    RAW_PATH,
    RAW_TIMESTAMP_FORMAT,
    TRAIN_FRAC,
    VAL_FRAC,
)
from ml.features import HORIZONS, TARGET_GASES, add_multi_horizon_targets, build_features

COLUMN_MAP = {
    "CO(GT)": "co_gt",
    "PT08.S1(CO)": "pt08_s1_co",
    "C6H6(GT)": "c6h6_gt",
    "PT08.S2(NMHC)": "pt08_s2_nmhc",
    "NOx(GT)": "nox_gt",
    "PT08.S3(NOx)": "pt08_s3_nox",
    "NO2(GT)": "no2_gt",
    "PT08.S4(NO2)": "pt08_s4_no2",
    "PT08.S5(O3)": "pt08_s5_o3",
    "T": "temp",
    "RH": "rh",
    "AH": "ah",
}
NUMERIC_COLS = list(COLUMN_MAP.values())

# Rows dropped at the end of train/val so no target (up to t+6h) reaches into the next split.
PURGE_ROWS = max(HORIZONS)


def parse_timestamps(df: pd.DataFrame) -> pd.Series:
    """Combine Date + Time with an explicit format. Raises instead of silently guessing."""
    time_clean = df["Time"].astype(str).str.replace(".", ":", regex=False)
    return pd.to_datetime(
        df["Date"].astype(str) + " " + time_clean,
        format=RAW_TIMESTAMP_FORMAT,
        errors="raise",
    )


def clean_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      clean_df : timestamp + 12 sensor columns on a complete hourly grid,
                 gaps filled CAUSALLY (forward-fill, max MAX_FFILL_HOURS).
      missing  : boolean mask (same rows) - True where a target gas was
                 originally missing (-200 or absent hour) before filling.
    """
    df = df.dropna(subset=["Date", "Time"]).copy()
    df["timestamp"] = parse_timestamps(df)
    df = df.drop(columns=["Date", "Time", "NMHC(GT)"], errors="ignore").rename(columns=COLUMN_MAP)

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found in raw data.")

    # Put every row on a strict 1-hour grid; missing hours become NaN rows.
    df = df.sort_values("timestamp").set_index("timestamp")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", name="timestamp")
    df = df.reindex(full_index)

    # UCI marks missing readings with -200.
    df[NUMERIC_COLS] = df[NUMERIC_COLS].replace(-200, np.nan)
    missing = df[TARGET_GASES].isna()

    # Causal fill only: past values flow forward, future values never flow backward.
    df[NUMERIC_COLS] = df[NUMERIC_COLS].ffill(limit=MAX_FFILL_HOURS)

    clean_df = df[NUMERIC_COLS].reset_index()
    validate_hourly(clean_df["timestamp"])
    return clean_df, missing.reset_index(drop=True)


def validate_hourly(ts: pd.Series) -> None:
    """Assert timestamps are strictly increasing and exactly 1 hour apart."""
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError("Timestamps are not strictly increasing.")
    steps = ts.diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise ValueError("Timestamps are not on an exact 1-hour grid.")


def mask_imputed_targets(feat_df: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    """Set a target to NaN if the future value it points at was originally missing."""
    out = feat_df.copy()
    for gas in TARGET_GASES:
        for h in HORIZONS:
            future_missing = missing[gas].shift(-h, fill_value=True).to_numpy()
            out.loc[future_missing, f"{gas}_target_{h}h"] = np.nan
    return out


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    purge: int = PURGE_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by row order (time), purging the last `purge` rows of train and val."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[: train_end - purge]
    val = df.iloc[train_end : val_end - purge]
    test = df.iloc[val_end:]
    return train, val, test


def build_split_frames(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Full pipeline up to (but not including) dropping NaN rows.
    Returns the cleaned hourly grid and the 3 purged splits, each still on a
    contiguous hourly grid (NaN where inputs/targets are invalid).
    Used by main() for the tabular CSVs and by ml.windowing for sequence tensors.
    """
    clean_df, missing = clean_raw_data(raw_df)
    feat_df = add_multi_horizon_targets(build_features(clean_df))
    feat_df = mask_imputed_targets(feat_df, missing)
    splits = dict(zip(["train", "val", "test"], chronological_split(feat_df)))
    return clean_df, {k: v.reset_index(drop=True) for k, v in splits.items()}


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_PATH}. Run `python -m ml.fetch_dataset` first.")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Cleaning raw data, building features + multi-horizon targets...")
    clean_df, splits = build_split_frames(pd.read_csv(RAW_PATH))
    print(f"   Hourly grid: {clean_df.shape} | {clean_df['timestamp'].min()} -> {clean_df['timestamp'].max()}")
    print(f"   Rows still NaN after causal fill (long gaps): {clean_df[NUMERIC_COLS].isna().any(axis=1).sum():,}")

    print(f"2. Chronological split (purge={PURGE_ROWS} rows at each boundary) -> tabular CSVs...")
    for name, part in splits.items():
        before = len(part)
        part = part.dropna().reset_index(drop=True)
        splits[name] = part
        part.to_csv(PROCESSED_DIR / f"{name}_features.csv", index=False)
        print(
            f"   - {name}_features.csv : {len(part):,} rows "
            f"(dropped {before - len(part):,} with gaps/imputed targets) "
            f"[{part['timestamp'].min()} -> {part['timestamp'].max()}]"
        )

    # Unseen test telemetry for the live simulator: real hours only (gaps act like packet loss).
    val_end = int(len(clean_df) * (TRAIN_FRAC + VAL_FRAC))
    test_stream_df = clean_df.iloc[val_end:].dropna().reset_index(drop=True)
    test_stream_df.to_csv(PROCESSED_DIR / "test_stream.csv", index=False)
    print(f"   - test_stream.csv    : {len(test_stream_df):,} rows (unseen telemetry for live replay)")


if __name__ == "__main__":
    main()
