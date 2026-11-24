from pathlib import Path
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_PATH = Path("data/raw/air_quality_raw.csv")


def main() -> None:
    print("Fetching UCI Air Quality Dataset (id=360)...")
    dataset = fetch_ucirepo(id=360)
    df = dataset.data.features.copy()

    # Save untouched raw dataset
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved raw dataset to: {RAW_PATH.resolve()}")

    # Quick audit
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # Count -200 sentinel values (UCI's missing-value marker)
    sentinel_counts = (df == -200).sum()
    missing_pct = ((df == -200) | df.isna()).mean() * 100

    audit = pd.DataFrame({
        "sentinel_-200_count": sentinel_counts,
        "total_missing_pct": missing_pct.round(2),
    })
    print("\nMissing Value Audit:")
    print(audit[audit["total_missing_pct"] > 0].to_string())


if __name__ == "__main__":
    main()