"""Leakage and correctness checks for Sub-Phase 1.1.

Unit tests use tiny synthetic data (always run).
Artifact tests check the real generated files (skipped if not generated yet).
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from ml.config import MAX_FFILL_HOURS, PROCESSED_DIR, RAW_PATH, SEQ_LEN, WINDOW_SCALERS_PATH, WINDOWS_PATH
from ml.features import HORIZONS, TARGET_GASES
from ml.preprocess import PURGE_ROWS, build_split_frames, chronological_split, clean_raw_data
from ml.windowing import TARGET_COLS, WINDOW_FEATURES, make_windows

RAW_COLS = ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)",
            "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
SPLITS = ["train", "val", "test"]


def _synthetic_raw(n: int = 12) -> pd.DataFrame:
    ts = pd.date_range("2004-03-10 18:00", periods=n, freq="h")
    df = pd.DataFrame({c: np.arange(n, dtype=float) + 1 for c in RAW_COLS})
    df.insert(0, "Date", ts.strftime("%-m/%-d/%Y"))
    df.insert(1, "Time", ts.strftime("%H:%M:%S"))
    return df


# ---------------- Unit tests (synthetic) ----------------

def test_causal_fill_never_uses_future_values():
    raw = _synthetic_raw()
    raw.loc[3, "CO(GT)"] = -200                      # short gap: 1h
    raw.loc[6:6 + MAX_FFILL_HOURS, "CO(GT)"] = -200  # long gap: MAX_FFILL_HOURS + 1 hours
    clean, missing = clean_raw_data(raw)
    co = clean["co_gt"]
    assert co[3] == co[2]                            # filled from the PAST, not interpolated
    assert co[6:6 + MAX_FFILL_HOURS].eq(co[5]).all()
    assert np.isnan(co[6 + MAX_FFILL_HOURS])         # beyond limit stays NaN
    assert missing.loc[3, "co_gt"] and not missing.loc[2, "co_gt"]


def test_missing_hours_are_reindexed_onto_hourly_grid():
    raw = _synthetic_raw().drop(index=5)              # an entire hour absent
    clean, missing = clean_raw_data(raw)
    assert len(clean) == 12
    assert (clean["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all()
    assert missing.loc[5].all()


def test_split_purges_boundary_rows():
    df = pd.DataFrame({"timestamp": pd.date_range("2004-01-01", periods=100, freq="h")})
    train, val, test = chronological_split(df, 0.7, 0.15, purge=6)
    assert train["timestamp"].max() + pd.Timedelta(hours=6) < val["timestamp"].min()
    assert val["timestamp"].max() + pd.Timedelta(hours=6) < test["timestamp"].min()


def test_make_windows_alignment_gaps_nans_and_mask():
    ts = pd.Series(pd.date_range("2004-01-01", periods=12, freq="h")).drop(index=5).reset_index(drop=True)
    X = np.arange(len(ts), dtype=float)[:, None]      # feature value == row index
    y = np.arange(len(ts), dtype=float)[:, None] * 10
    X[9, 0] = np.nan                                  # bad input -> windows containing row 9 dropped
    y[3, 0] = np.nan                                  # missing target -> kept but masked... unless all targets NaN
    y = np.hstack([y, y + 1])
    y[3, 1] = 5.0                                     # row 3 still has one real target
    X_w, y_w, m_w, end_ts = make_windows(X, y, ts, seq_len=3)
    # Hour gap after row 4; NaN at row 9 -> valid window ends: 2,3,4 (pre-gap) and 7,8 (rows 5..8 post-gap).
    assert X_w[:, -1, 0].tolist() == [2, 3, 4, 7, 8]  # last step == target row
    assert (np.diff(X_w[:, :, 0], axis=1) == 1).all()  # oldest -> newest order
    assert m_w[1].tolist() == [0.0, 1.0] and y_w[1, 0] == 0.0
    assert m_w[[0, 2, 3, 4]].all()
    assert np.isfinite(X_w).all() and np.isfinite(y_w).all()


# ---------------- Artifact tests (real data) ----------------

needs_artifacts = pytest.mark.skipif(
    not WINDOWS_PATH.exists(), reason="Run ml.preprocess and ml.windowing first."
)


@pytest.fixture(scope="module")
def csv_splits():
    out = {}
    for name in SPLITS:
        df = pd.read_csv(PROCESSED_DIR / f"{name}_features.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        out[name] = df
    return out


@pytest.fixture(scope="module")
def grid_splits():
    return build_split_frames(pd.read_csv(RAW_PATH))[1]


@pytest.fixture(scope="module")
def raw_missing():
    clean, missing = clean_raw_data(pd.read_csv(RAW_PATH))
    return missing.set_index(clean["timestamp"])


@needs_artifacts
def test_csv_splits_are_chronological_purged_and_clean(csv_splits):
    gap = pd.Timedelta(hours=PURGE_ROWS)
    for df in csv_splits.values():
        assert df["timestamp"].is_monotonic_increasing and not df["timestamp"].duplicated().any()
        assert not df.isna().any().any()
    assert csv_splits["train"]["timestamp"].max() + gap < csv_splits["val"]["timestamp"].min()
    assert csv_splits["val"]["timestamp"].max() + gap < csv_splits["test"]["timestamp"].min()


@needs_artifacts
def test_no_csv_target_comes_from_imputed_value(csv_splits, raw_missing):
    for df in csv_splits.values():
        for gas in TARGET_GASES:
            for h in HORIZONS:
                future_ts = df["timestamp"] + pd.Timedelta(hours=h)
                assert not raw_missing.loc[future_ts, gas].any(), f"{gas} t+{h}h uses imputed value"


@needs_artifacts
def test_window_tensors_shapes_finite_aligned_and_masked(grid_splits, raw_missing):
    data = np.load(WINDOWS_PATH)
    scalers = joblib.load(WINDOW_SCALERS_PATH)
    for name in SPLITS:
        X, y, m, ts = (data[f"{k}_{name}"] for k in ["X", "y", "m", "ts"])
        assert X.ndim == 3 and X.shape[1:] == (SEQ_LEN, len(WINDOW_FEATURES)) and len(X) > 0
        assert y.shape == m.shape == (len(X), len(TARGET_COLS)) and len(ts) == len(X)
        assert np.isfinite(X).all() and np.isfinite(y).all()

        rows = grid_splits[name].set_index("timestamp").loc[pd.to_datetime(ts)]
        # Last timestep of each window == the row whose targets we predict.
        np.testing.assert_allclose(
            scalers["x_scaler"].inverse_transform(X[:, -1, :]), rows[WINDOW_FEATURES].to_numpy(), rtol=1e-4, atol=1e-3
        )
        # Real targets match; masked targets are exactly those whose future hour was originally missing.
        y_true = rows[TARGET_COLS].to_numpy()
        valid = m.astype(bool)
        assert (valid == np.isfinite(y_true)).all()
        y_inv = scalers["y_scaler"].inverse_transform(y)
        np.testing.assert_allclose(y_inv[valid], y_true[valid], rtol=1e-4, atol=1e-3)
        for j, col in enumerate(TARGET_COLS):
            gas, h = col.split("_target_")[0], int(col.split("_")[-1][:-1])
            future_ts = pd.to_datetime(ts) + pd.Timedelta(hours=h)
            future_missing = raw_missing[gas].reindex(future_ts, fill_value=True).to_numpy()
            assert not (valid[:, j] & future_missing).any()

        # Windows never reach back across a split boundary.
        assert pd.Timestamp(ts.min()) >= grid_splits[name]["timestamp"].min() + pd.Timedelta(hours=SEQ_LEN - 1)

    assert pd.Timestamp(data["ts_train"].max()) + pd.Timedelta(hours=PURGE_ROWS) < pd.Timestamp(data["ts_val"].min())
    assert pd.Timestamp(data["ts_val"].max()) + pd.Timedelta(hours=PURGE_ROWS) < pd.Timestamp(data["ts_test"].min())


@needs_artifacts
def test_scalers_fit_on_train_only(grid_splits):
    scalers = joblib.load(WINDOW_SCALERS_PATH)
    train_mean = grid_splits["train"][WINDOW_FEATURES].mean().to_numpy()
    all_mean = pd.concat(grid_splits.values())[WINDOW_FEATURES].mean().to_numpy()
    np.testing.assert_allclose(scalers["x_scaler"].mean_, train_mean, rtol=1e-6)
    assert not np.allclose(scalers["x_scaler"].mean_, all_mean, rtol=1e-3)
