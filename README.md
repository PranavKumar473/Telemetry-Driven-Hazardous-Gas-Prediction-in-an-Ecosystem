# Telemetry-Driven Hazardous Gas Prediction in an Ecosystem

A simulated real-time environmental telemetry system that ingests multivariate gas and weather streams, forecasts hazardous concentration spikes (`1h`, `3h`, and `6h` ahead) using **XGBoost** and a **PyTorch LSTM**, detects abnormal sensor signatures with **Isolation Forest**, and serves live alerts and model benchmarks via **FastAPI**, **SQLite (Raw SQL)**, and **Streamlit**.

---

## Architecture

```text
[Telemetry Simulator] --(HTTP POST)--> [FastAPI Ingestion Service]
                                               |
              +--------------------------------+--------------------------------+
              |                                                                 |
              v                                                                 v
   [SQLite Database (Raw SQL)]                                    [Sliding-Window ML Engine]
   (Indexed time-series storage)                               (XGBoost + PyTorch LSTM + IsoForest)
              |                                                                 |
              +--------------------------------+--------------------------------+
                                               |
                                               v
                                 [Streamlit Live Dashboard]
                          (Real-time plots, alert feed, benchmark)
```

---

## Project Structure

```text
hazardous-gas-telemetry/
├── data/
│   ├── raw/                  # Untouched UCI Air Quality dataset
│   ├── processed/            # Chronological train/val/test splits & test_stream.csv
│   └── telemetry.db          # SQLite database (auto-created on backend startup)
├── models/                   # Saved XGBoost, PyTorch LSTM, Isolation Forest & benchmark JSON
├── ml/
│   ├── fetch_dataset.py      # Phase 0: Downloads UCI Air Quality dataset (id=360)
│   ├── features.py           # Phase 1: Lag, rolling, derivative, and cyclical features
│   ├── preprocess.py         # Phase 1: Cleaning (-200 imputation) & 70/15/15 chronological split
│   ├── train_xgboost.py      # Phase 2: Multi-horizon XGBoost + Isolation Forest training
│   ├── train_lstm.py         # Phase 2: PyTorch LSTM sequence model training
│   └── evaluate.py           # Phase 2: Offline MAE, RMSE, Breach Recall, and latency benchmark
├── backend/
│   ├── db.py                 # Raw SQLite schema, indexed queries, and inserts
│   ├── alert_engine.py       # Tiered threshold evaluation (SAFE/ELEVATED/WARNING/CRITICAL)
│   ├── inference.py          # Sliding-window live inference engine
│   └── main.py               # FastAPI server (/ingest, /history, /benchmark, /reset)
├── simulator/
│   ├── fault_injector.py     # Synthetic leak, drift, and packet-loss fault modes
│   └── producer.py           # Replays unseen test_stream.csv as live telemetry packets
├── frontend/
│   └── app.py                # Streamlit live monitoring & offline benchmark dashboard
└── requirements.txt
```

---

## 1. Environment Setup (`cmd`)

Run these commands from the `hazardous-gas-telemetry` root directory:

```cmd
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Offline Data & ML Pipeline (`cmd`)

Run these scripts once in order before starting the live system (with `venv` activated):

### Phase 0 — Download & Audit Dataset
```cmd
python ml\fetch_dataset.py
```

### Phase 1 — Clean, Split & Engineer Features
```cmd
python -m ml.preprocess
```

### Phase 2 — Train Models & Generate Benchmark Report
```cmd
python -m ml.train_xgboost
python -m ml.train_lstm
python -m ml.evaluate
```

---

## 3. Running the Live System (`cmd`)

Open **three separate Command Prompt windows** in the `hazardous-gas-telemetry` root directory and activate the virtual environment in each:

```cmd
venv\Scripts\activate.bat
```

### Terminal 1 — Serve the FastAPI Backend
```cmd
uvicorn backend.main:app --reload
```
* **API Base URL:** `http://127.0.0.1:8000`
* **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`

### Terminal 2 — Serve the Streamlit Frontend
```cmd
streamlit run frontend\app.py
```
* **Dashboard URL:** `http://localhost:8501`

### Terminal 3 — Run the Telemetry Simulator
Choose a fault-injection mode to stream unseen test data into the backend:

```cmd
REM Normal historical replay
python -m simulator.producer --mode normal --interval 1.0 --limit 100

REM Exponential gas leak ramp starting at step 20 (triggers early warnings)
python -m simulator.producer --mode leak --interval 1.0 --limit 100

REM Gradual upward sensor drift
python -m simulator.producer --mode drift --interval 1.0 --limit 100

REM Simulated 15% random telemetry packet loss
python -m simulator.producer --mode packet_loss --interval 1.0 --limit 100
```

*(Tip: Click **Reset Telemetry Database** in the Streamlit sidebar between simulation runs to start with a clean chart.)*

---

## 4. API Endpoints