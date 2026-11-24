"""Central paths and pipeline constants for the ML data pipeline.

Every path is anchored to the project root (derived from this file's
location), so scripts work no matter which directory you launch them from.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "air_quality_raw.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

WINDOWS_PATH = PROCESSED_DIR / "windows.npz"
WINDOW_SCALERS_PATH = MODELS_DIR / "window_scalers.joblib"

# Raw UCI timestamp format after "Date Time" concatenation (e.g. "3/10/2004 18:00:00").
RAW_TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

# Chronological split fractions (test gets the remaining 15%).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Causal gap filling: only carry the last real reading forward this many hours.
# Longer gaps stay NaN and the affected rows are dropped (no fake flat lines).
MAX_FFILL_HOURS = 3

# Sliding-window length for sequence models: 24h = one full daily cycle.
SEQ_LEN = 24
