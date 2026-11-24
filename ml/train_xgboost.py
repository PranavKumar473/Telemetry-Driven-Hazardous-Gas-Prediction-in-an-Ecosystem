import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
from ml.config import MODELS_DIR, PROCESSED_DIR
from ml.features import HORIZONS, TARGET_GASES, get_feature_columns


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_features.csv")

    feature_cols = get_feature_columns(train_df)
    X_train, X_val = train_df[feature_cols], val_df[feature_cols]

    # Save feature column order for live inference
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Training XGBoost on {len(train_df):,} train / {len(val_df):,} val rows x {len(feature_cols)} features...")
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
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            out_path = MODELS_DIR / f"xgb_{gas}_{h}h.json"
            model.save_model(out_path)
            val_mae = float(np.mean(np.abs(model.predict(X_val) - y_val)))
            print(f"  {out_path.name:<20} best_iter={model.best_iteration:<4} val MAE={val_mae:.3f}")

    print("\nTraining Isolation Forest anomaly detector (train split only)...")
    iso = IsolationForest(n_estimators=150, contamination=0.02, random_state=42, n_jobs=-1)
    iso.fit(X_train)
    joblib.dump(iso, MODELS_DIR / "isolation_forest.joblib")
    val_flag_rate = float((iso.predict(X_val) == -1).mean())
    print(f"  Saved isolation_forest.joblib (val anomaly rate={val_flag_rate:.1%})")


if __name__ == "__main__":
    main()
