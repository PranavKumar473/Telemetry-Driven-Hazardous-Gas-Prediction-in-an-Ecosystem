from pathlib import Path
import numpy as np
import pandas as pd
from ml.features import build_features, add_multi_horizon_targets

RAW_PATH = Path("data/raw/air_quality_raw.csv")
PROCESSED_DIR = Path("data/processed")

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


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Combine Date and Time into a single chronological timestamp
    time_clean = df["Time"].astype(str).str.replace(".", ":", regex=False)
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + time_clean,
        format="mixed",
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Drop ~90% missing NMHC(GT) and raw Date/Time strings
    df = df.drop(columns=["Date", "Time", "NMHC(GT)"], errors="ignore")
    df = df.rename(columns=COLUMN_MAP)

    # Replace UCI -200 sentinel values with NaN
    numeric_cols = list(COLUMN_MAP.values())
    df[numeric_cols] = df[numeric_cols].replace(-200, np.nan)

    # Chronological imputation: interpolate short gaps (<=6h), then forward-fill
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=6)
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Reorder columns with timestamp first
    return df[["timestamp"] + numeric_cols]


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_PATH}. Run ml/fetch_dataset.py first.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading and cleaning raw dataset...")
    raw_df = pd.read_csv(RAW_PATH)
    clean_df = clean_raw_data(raw_df)
    print(f"   Cleaned shape: {clean_df.shape}")

    # Strict chronological 70 / 15 / 15 split
    n = len(clean_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Save clean unseen test telemetry for the Phase 3 live simulator
    test_stream_df = clean_df.iloc[val_end:].reset_index(drop=True)
    test_stream_df.to_csv(PROCESSED_DIR / "test_stream.csv", index=False)

    print("2. Building features and multi-horizon targets...")
    feat_df = build_features(clean_df)
    feat_df = add_multi_horizon_targets(feat_df)

    # Slice by the exact same chronological indices, then drop boundary NaNs
    train_df = feat_df.iloc[:train_end].dropna().reset_index(drop=True)
    val_df = feat_df.iloc[train_end:val_end].dropna().reset_index(drop=True)
    test_df = feat_df.iloc[val_end:].dropna().reset_index(drop=True)

    train_df.to_csv(PROCESSED_DIR / "train_features.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val_features.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_features.csv", index=False)

    print("\nSaved processed splits to data/processed/:")
    print(f"  - train_features.csv : {len(train_df):,} rows ({train_df['timestamp'].min()} -> {train_df['timestamp'].max()})")
    print(f"  - val_features.csv   : {len(val_df):,} rows ({val_df['timestamp'].min()} -> {val_df['timestamp'].max()})")
    print(f"  - test_features.csv  : {len(test_df):,} rows ({test_df['timestamp'].min()} -> {test_df['timestamp'].max()})")
    print(f"  - test_stream.csv    : {len(test_stream_df):,} rows (unseen telemetry for live replay)")


if __name__ == "__main__":
    main()