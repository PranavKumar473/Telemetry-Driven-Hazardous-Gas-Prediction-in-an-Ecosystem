"""Raw SQLite storage layer for live telemetry, multi-model predictions, and alerts."""
import sqlite3
from pathlib import Path
from typing import Any
from ml.config import DATA_DIR, SEQ_LEN

DB_PATH: Path = DATA_DIR / "telemetry.db"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            co_gt REAL NOT NULL,
            pt08_s1_co REAL NOT NULL,
            c6h6_gt REAL NOT NULL,
            pt08_s2_nmhc REAL NOT NULL,
            nox_gt REAL NOT NULL,
            pt08_s3_nox REAL NOT NULL,
            no2_gt REAL NOT NULL,
            pt08_s4_no2 REAL NOT NULL,
            pt08_s5_o3 REAL NOT NULL,
            temp REAL NOT NULL,
            rh REAL NOT NULL,
            ah REAL NOT NULL,
            fault_mode TEXT NOT NULL DEFAULT 'normal'
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_telemetry_node_id_desc
        ON telemetry_raw (node_id, id DESC);
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            co_pred_1h REAL NOT NULL,
            co_pred_3h REAL NOT NULL,
            co_pred_6h REAL NOT NULL,
            no2_pred_1h REAL NOT NULL,
            no2_pred_3h REAL NOT NULL,
            no2_pred_6h REAL NOT NULL,
            c6h6_pred_1h REAL NOT NULL,
            c6h6_pred_3h REAL NOT NULL,
            c6h6_pred_6h REAL NOT NULL,
            is_anomaly INTEGER NOT NULL DEFAULT 0,
            latency_ms REAL NOT NULL
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_node_id_desc
        ON predictions (node_id, id DESC);
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            severity TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'GAS_BREACH',
            gas TEXT NOT NULL,
            message TEXT NOT NULL
        );
    """)

    # Lightweight migration if an older alerts table existed without 'category'
    cur.execute("PRAGMA table_info(alerts);")
    cols = {row["name"] for row in cur.fetchall()}
    if "category" not in cols:
        cur.execute("ALTER TABLE alerts ADD COLUMN category TEXT NOT NULL DEFAULT 'GAS_BREACH';")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_node_id_desc
        ON alerts (node_id, id DESC);
    """)

    conn.commit()
    conn.close()


def insert_telemetry(payload: dict[str, Any], db_path: Path | None = None) -> int:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO telemetry_raw (
            node_id, recorded_at, co_gt, pt08_s1_co, c6h6_gt, pt08_s2_nmhc,
            nox_gt, pt08_s3_nox, no2_gt, pt08_s4_no2, pt08_s5_o3,
            temp, rh, ah, fault_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            payload["node_id"],
            payload["timestamp"],
            float(payload["co_gt"]),
            float(payload["pt08_s1_co"]),
            float(payload["c6h6_gt"]),
            float(payload["pt08_s2_nmhc"]),
            float(payload["nox_gt"]),
            float(payload["pt08_s3_nox"]),
            float(payload["no2_gt"]),
            float(payload["pt08_s4_no2"]),
            float(payload["pt08_s5_o3"]),
            float(payload["temp"]),
            float(payload["rh"]),
            float(payload["ah"]),
            payload.get("fault_mode", "normal"),
        ),
    )
    row_id = int(cur.lastrowid)
    conn.commit()
    conn.close()
    return row_id


def fetch_recent_window(node_id: str, limit: int = SEQ_LEN, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Fetch the most recent `limit` telemetry readings for `node_id`, ordered oldest -> newest."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT recorded_at AS timestamp, co_gt, pt08_s1_co, c6h6_gt, pt08_s2_nmhc,
               nox_gt, pt08_s3_nox, no2_gt, pt08_s4_no2, pt08_s5_o3, temp, rh, ah
        FROM telemetry_raw
        WHERE node_id = ?
        ORDER BY id DESC
        LIMIT ?;
        """,
        (node_id, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    rows.reverse()  # Chronological order (oldest -> newest)
    return rows


def insert_prediction(record: dict[str, Any], db_path: Path | None = None) -> None:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions (
            node_id, recorded_at, model_name,
            co_pred_1h, co_pred_3h, co_pred_6h,
            no2_pred_1h, no2_pred_3h, no2_pred_6h,
            c6h6_pred_1h, c6h6_pred_3h, c6h6_pred_6h,
            is_anomaly, latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            record["node_id"],
            record["recorded_at"],
            record["model_name"],
            float(record["co_pred_1h"]),
            float(record["co_pred_3h"]),
            float(record["co_pred_6h"]),
            float(record["no2_pred_1h"]),
            float(record["no2_pred_3h"]),
            float(record["no2_pred_6h"]),
            float(record["c6h6_pred_1h"]),
            float(record["c6h6_pred_3h"]),
            float(record["c6h6_pred_6h"]),
            int(bool(record["is_anomaly"])),
            float(record["latency_ms"]),
        ),
    )
    conn.commit()
    conn.close()


def insert_alerts(node_id: str, recorded_at: str, alerts: list[dict[str, str]], db_path: Path | None = None) -> None:
    if not alerts:
        return
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO alerts (node_id, recorded_at, severity, category, gas, message)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        [
            (
                node_id,
                recorded_at,
                a["severity"],
                a.get("category", "GAS_BREACH"),
                a["gas"],
                a["message"],
            )
            for a in alerts
        ],
    )
    conn.commit()
    conn.close()


def fetch_history(node_id: str = "eco_node_01", limit: int = 120, db_path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """
    Return recent telemetry, predictions, and alerts for `node_id`.
    Telemetry and predictions are returned oldest -> newest for time-series charting;
    alerts are returned newest -> oldest for the live incident feed.
    """
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, node_id, recorded_at, co_gt, pt08_s1_co, c6h6_gt, pt08_s2_nmhc,
               nox_gt, pt08_s3_nox, no2_gt, pt08_s4_no2, pt08_s5_o3,
               temp, rh, ah, fault_mode
        FROM telemetry_raw
        WHERE node_id = ?
        ORDER BY id DESC
        LIMIT ?;
        """,
        (node_id, limit),
    )
    telemetry = [dict(r) for r in cur.fetchall()]
    telemetry.reverse()

    cur.execute(
        """
        SELECT id, node_id, recorded_at, model_name,
               co_pred_1h, co_pred_3h, co_pred_6h,
               no2_pred_1h, no2_pred_3h, no2_pred_6h,
               c6h6_pred_1h, c6h6_pred_3h, c6h6_pred_6h,
               is_anomaly, latency_ms
        FROM predictions
        WHERE node_id = ?
        ORDER BY id DESC
        LIMIT ?;
        """,
        (node_id, limit * 3),
    )
    predictions = [dict(r) for r in cur.fetchall()]
    predictions.reverse()

    cur.execute(
        """
        SELECT id, node_id, recorded_at, severity, category, gas, message
        FROM alerts
        WHERE node_id = ?
        ORDER BY id DESC
        LIMIT ?;
        """,
        (node_id, limit),
    )
    alerts = [dict(r) for r in cur.fetchall()]

    conn.close()
    return {
        "telemetry": telemetry,
        "predictions": predictions,
        "alerts": alerts,
    }


def reset_db(db_path: Path | None = None) -> None:
    """Clear all runtime telemetry, predictions, and alerts tables."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM telemetry_raw;")
    cur.execute("DELETE FROM predictions;")
    cur.execute("DELETE FROM alerts;")
    conn.commit()
    conn.close()
