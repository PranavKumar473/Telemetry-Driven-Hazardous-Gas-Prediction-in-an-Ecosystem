import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
from ml.features import TARGET_GASES, HORIZONS, get_feature_columns

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_features.csv")

    feature_cols = get_feature_columns(train_df)
    X_train, X_val = train_df[feature_cols], val_df[feature_cols]

    # Save feature column order for live inference
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Training XGBoost models on {len(feature_cols)} features...")
    for gas in TARGET_GASES:
        for h in HORIZONS:
            target_col = f"{gas}_target_{h}h"
            y_train, y_val = train_df[target_col], val_df[target_col]

            model = XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                early_stopping_rounds=25,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            out_path = MODELS_DIR / f"xgb_{gas}_{h}h.json"
            model.save_model(out_path)
            print(f"  Saved {out_path.name} (best_iteration={model.best_iteration})")

    print("\nTraining Isolation Forest anomaly detector...")
    iso = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)
    joblib.dump(iso, MODELS_DIR / "isolation_forest.joblib")
    print("  Saved isolation_forest.joblib")


if __name__ == "__main__":
    main()