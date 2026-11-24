# Telemetry-Driven Hazardous Gas Prediction in an Ecosystem

A real-time environmental telemetry and multi-horizon hazardous gas forecasting system that ingests multivariate gas and weather sensor streams, forecasts hazardous concentration spikes (**1h**, **3h**, and **6h** ahead) for **Carbon Monoxide ($\text{CO}$)**, **Nitrogen Dioxide ($\text{NO}_2$)**, and **Benzene ($\text{C}_6\text{H}_6$)** using **XGBoost**, a **PyTorch Stacked LSTM**, and a **1D-CNN + LSTM Hybrid**, detects abnormal multivariate sensor signatures with **Isolation Forest**, and serves cooldown-protected tiered alerts via **FastAPI**, **SQLite (Raw SQL)**, and **Streamlit**.

---

## Architecture

```text
[Telemetry Simulator] --(HTTP POST /ingest)--> [FastAPI Ingestion & Validation Service]
 (normal / leak / drift / packet_loss)                        |
               +----------------------------------------------+----------------------------------------------+
               |                                              |                                              |
               v                                              v                                              v
   [SQLite Database (Raw SQL)]                 [Dual-Path Sliding-Window ML Engine]             [Stateful Alert Engine]
  (Indexed telemetry, predictions,            - Tabular (39 features): XGBoost + IsoForest     - Tiered: SAFE/ELEVATED/WARN/CRIT
   and categorized alert tables)              - Sequence (1, 24, 16): GasLSTM + 1D-CNN+LSTM    - Cooldown & 90% hysteresis
               |                                              |                                              |
               +----------------------------------------------+----------------------------------------------+
                                                              |
                                                              v
                                               [Streamlit Live Dashboard]
                                 (3-model forecast chart, split alert feed, benchmark tab)
```

---

## Project Structure

```text
telemetry-driven-hazardous-gas-prediction-in-ecosystem/
├── data/
│   ├── raw/                      # Untouched UCI Air Quality dataset (id=360)
│   ├── processed/                # Leakage-free train/val/test CSVs, windows.npz & test_stream.csv
│   └── telemetry.db              # SQLite database (auto-created on backend startup)
├── models/                       # Trained XGBoost, GasLSTM, CNNLSTM, IsoForest, scalers & benchmark JSON
├── ml/
│   ├── config.py                 # Project-root-anchored paths and pipeline constants (SEQ_LEN=24)
│   ├── fetch_dataset.py          # Downloads UCI Air Quality dataset (id=360) and audits -200 sentinels
│   ├── features.py               # Cyclical time, 1h/3h/6h lags, 6h rolling stats, and 1h/3h derivatives
│   ├── preprocess.py             # Strict hourly grid, causal ffill (<=3h), target masking & 6h boundary purge
│   ├── windowing.py              # Train-only StandardScaler + (N, 24, 16) sliding-window tensor builder
│   ├── models.py                 # PyTorch GasLSTM (Unit IV) and CNNLSTM (Units II-IV) architectures
│   ├── nn_training.py            # Masked Huber loss, shuffled train batches & early-stopping trainer
│   ├── train_xgboost.py          # 9 multi-horizon XGBoost regressors + Isolation Forest anomaly detector
│   ├── train_lstm.py             # Trains PyTorch GasLSTM on windows.npz
│   ├── train_cnn_lstm.py         # Trains PyTorch 1D-CNN + LSTM hybrid on windows.npz
│   └── evaluate.py               # Unified physical-unit test benchmark (MAE, RMSE, Breach Recall, latency)
├── backend/
│   ├── db.py                     # Raw SQLite schema, composite indexes, window queries, history & reset
│   ├── inference.py              # Startup model loader & dual-path (39-feat tabular + 16-feat sequence) engine
│   ├── alert_engine.py           # Stateful AlertCooldownManager separating GAS_BREACH vs. SENSOR_ANOMALY
│   └── main.py                   # FastAPI service with strict Pydantic v2 validation (/ingest, /history, etc.)
├── simulator/
│   ├── fault_injector.py         # Synthetic leak, drift, and packet-loss fault injection modes
│   └── producer.py               # Streams unseen test_stream.csv packets into FastAPI /ingest
├── frontend/
│   └── app.py                    # Streamlit live 3-model dashboard & offline benchmark viewer
├── tests/
│   ├── test_data_pipeline.py     # Leakage, causal imputation, boundary purge & window alignment checks
│   ├── test_models.py            # Neural network shape/finiteness, masked loss & checkpoint tests
│   ├── test_backend.py           # FastAPI TestClient, warm-up, alert cooldown & Pydantic v2 tests
│   └── test_e2e.py               # End-to-end 30-packet normal, leak & packet-loss simulation tests
└── requirements.txt
```

---

## 1. Environment Setup

### macOS / Linux (`zsh` / `bash`)
```bash
# On macOS (Apple Silicon / Intel), install OpenMP runtime for XGBoost once:
brew install libomp

python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (`cmd`)
```cmd
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Offline Data & ML Pipeline

With your virtual environment activated, run the pipeline in order from the repository root:

```bash
# 1. Download raw UCI Air Quality dataset (9,357 hourly rows)
python -m ml.fetch_dataset

# 2. Clean causally, split 70/15/15 with 6-hour boundary purge, and build (N, 24, 16) window tensors
python -m ml.preprocess
python -m ml.windowing

# 3. Train all three forecasting model families + Isolation Forest anomaly detector
python -m ml.train_xgboost
python -m ml.train_lstm
python -m ml.train_cnn_lstm

# 4. Run unified physical-unit evaluation on the 1,067-timestamp common test set
python -m ml.evaluate

# 5. Run the automated verification test suite (21 tests)
pytest -v
```

---

## 3. Running the Live Telemetry System

Open **three terminals** in the project root with `venv` activated (`source venv/bin/activate` on macOS/Linux or `venv\Scripts\activate.bat` on Windows):

### Terminal 1 — FastAPI Backend
```bash
uvicorn backend.main:app --reload
```
* **API Base URL:** `http://127.0.0.1:8000`
* **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`

### Terminal 2 — Streamlit Live Dashboard
```bash
streamlit run frontend/app.py
```
* **Dashboard URL:** `http://localhost:8501`

### Terminal 3 — Telemetry Simulator
```bash
# Normal historical replay (unseen test split)
python -m simulator.producer --mode normal --interval 0.5 --limit 100

# Exponential gas leak ramp starting at step 20 (triggers WARNING -> CRITICAL alerts)
python -m simulator.producer --mode leak --interval 0.5 --limit 100

# Gradual upward sensor calibration drift across CO, NO2, and C6H6
python -m simulator.producer --mode drift --interval 0.5 --limit 100

# Simulated 15% random telemetry packet loss
python -m simulator.producer --mode packet_loss --interval 0.5 --limit 100
```

---

## 4. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns backend health status, loaded model families (`XGBoost`, `IsolationForest`, `PyTorch_LSTM`, `CNN_LSTM`), and `seq_len=24`. |
| `POST` | `/ingest` | Validates a `TelemetryPacket` (`ConfigDict(extra="forbid")`), stores it in SQLite, runs sliding-window inference (`WARMING_UP` for packets 1–6, `XGBoost` at $\ge 7$, all 3 models at $\ge 24$), evaluates cooldown-protected alerts, and returns forecasts + triggered alerts. |
| `GET` | `/history` | Query params: `node_id` (default `eco_node_01`), `limit` (1–1000). Returns chronological `telemetry` and `predictions` arrays plus newest-first `alerts`. |
| `GET` | `/benchmark` | Returns the 27-row offline test benchmark (`models/benchmark_report.json`). |
| `POST` | `/reset` | Truncates `telemetry_raw`, `predictions`, and `alerts` tables and resets in-memory alert cooldown state. |

---

## 5. Phase 1 Offline Test Benchmark Results (Unseen Test Split, 1,067 Common Timestamps)

All predictions are evaluated in physical units ($\text{mg/m}^3$ for $\text{CO}$, $\mu\text{g/m}^3$ for $\text{NO}_2$ and $\text{C}_6\text{H}_6$) exclusively on non-imputed ground-truth targets (`m_test == 1`):

| Model | Parameters | Mean Hazardous Breach Recall | Median Single-Packet Latency | CO Mean MAE ($\text{mg/m}^3$) | $\text{NO}_2$ Mean MAE ($\mu\text{g/m}^3$) | $\text{C}_6\text{H}_6$ Mean MAE ($\mu\text{g/m}^3$) |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** (9 Regressors, 39 tabular features) | 9 boosted trees | 0.593 | 13.225 ms | **0.660** | **22.941** | 3.709 |
| **PyTorch Stacked LSTM** (`(1, 24, 16)` window) | 54,857 | **0.720** | 0.319 ms | 0.722 | 27.211 | 4.048 |
| **PyTorch 1D-CNN + LSTM** (`(1, 24, 16)` window) | **30,345** | 0.714 | **0.129 ms** | 0.698 | 26.439 | **3.576** |
