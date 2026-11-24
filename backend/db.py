import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("data/telemetry.db")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            co_gt REAL,
            pt08_s1_co REAL,
            c6h6_gt REAL,
            pt08_s2_nmhc REAL,
            nox_gt REAL,
            pt08_s3_nox REAL,
            no2_gt REAL,
            pt08_s4_no2 REAL,
            pt08_s5_o3 REAL,
            temp REAL,
            rh REAL,
            ah REAL,
            fault_mode TEXT DEFAULT 'normal'
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_telemetry_node_time
        ON telemetry_raw (node_id, recorded_at);
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            co_pred_1h REAL,
            co_pred_3h REAL,
            co_pred_6h REAL,
            no2_pred_1h REAL,
            no2_pred_3h REAL,
            no2_pred_6h REAL,
            c6h6_pred_1h REAL,
            c6h6_pred_3h REAL,
            c6h6_pred_6h REAL,
            is_anomaly INTEGER DEFAULT 0,
            latency_ms REAL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL,
            severity TEXT NOT NULL,
            gas TEXT NOT NULL,
            message TEXT NOT NULL
        );
    """)

    conn.commit()
    conn.close()


def insert_telemetry(payload: dict[str, Any]) -> int:
    conn = get_connection()
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
            payload["node_id"], payload["timestamp"], payload["co_gt"],
            payload["pt08_s1_co"], payload["c6h6_gt"], payload["pt08_s2_nmhc"],
            payload["nox_gt"], payload["pt08_s3_nox"], payload["no2_gt"],
            payload["pt08_s4_no2"], payload["pt08_s5_o3"], payload["temp"],
            payload["rh"], payload["ah"], payload.get("fault_mode", "normal"),
        ),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def fetch_recent_window(node_id: str, limit: int = 24) -> list[dict[str, Any]]:
    conn = get_connection()
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


def insert_prediction(record: dict[str, Any]) -> None:
    conn = get_connection()
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
            record["node_id"], record["recorded_at"], record["model_name"],
            record["co_pred_1h"], record["co_pred_3h"], record["co_pred_6h"],
            record["no2_pred_1h"], record["no2_pred_3h"], record["no2_pred_6h"],
            record["c6h6_pred_1h"], record["c6h6_pred_3h"], record["c6h6_pred_6h"],
            int(record["is_anomaly"]), record["latency_ms"],
        ),
    )
    conn.commit()
    conn.close()


def insert_alerts(node_id: str, recorded_at: str, alerts: list[dict[str, str]]) -> None:
    if not alerts:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO alerts (node_id, recorded_at, severity, gas, message)
        VALUES (?, ?, ?, ?, ?);
        """,
        [(node_id, recorded_at, a["severity"], a["gas"], a["message"]) for a in alerts],
    )
    conn.commit()
    conn.close()


def fetch_history(node_id: str = "eco_node_01", limit: int = 120) -> dict[str, list[dict]]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT * FROM telemetry_raw
        WHERE node_id =