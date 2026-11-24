"""Live telemetry simulator that replays unseen test_stream.csv packets into FastAPI /ingest."""
import argparse
import random
import time
from typing import Any
import pandas as pd
import requests
from ml.config import PROCESSED_DIR
from ml.features import SENSOR_COLS
from simulator.fault_injector import VALID_MODES, apply_fault

STREAM_PATH = PROCESSED_DIR / "test_stream.csv"
API_URL = "http://127.0.0.1:8000/ingest"


def build_packet(
    row: dict[str, Any],
    node_id: str,
    step: int,
    mode: str = "normal",
    rng: random.Random | None = None,
) -> dict[str, Any] | None:
    """
    Format a single telemetry row into the strict 15-field schema expected by
    backend.main.TelemetryPacket (ConfigDict(extra='forbid')), applying any fault injection.
    """
    base: dict[str, Any] = {
        "node_id": str(node_id),
        "timestamp": str(row["timestamp"]),
        **{col: float(row[col]) for col in SENSOR_COLS},
        "fault_mode": mode,
    }
    return apply_fault(base, step=step, mode=mode, rng=rng)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Telemetry Simulator")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="normal")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between packets")
    parser.add_argument("--limit", type=int, default=150, help="Max packets to replay")
    parser.add_argument("--node", type=str, default="eco_node_01")
    parser.add_argument("--url", type=str, default=API_URL, help="FastAPI /ingest URL")
    args = parser.parse_args()

    if not STREAM_PATH.exists():
        raise FileNotFoundError(f"Missing {STREAM_PATH}. Run `python -m ml.preprocess` first.")

    df = pd.read_csv(STREAM_PATH).head(args.limit)
    print(f"Streaming {len(df)} packets | node={args.node} | mode={args.mode} | interval={args.interval}s")

    for step, (_, row) in enumerate(df.iterrows(), start=1):
        payload = build_packet(row.to_dict(), node_id=args.node, step=step, mode=args.mode)

        if payload is None:
            print(f"[Step {step:03d}] DROPPED PACKET (simulated packet_loss)")
            time.sleep(args.interval)
            continue

        try:
            resp = requests.post(args.url, json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "UNKNOWN")
            alert_cnt = len(data.get("alerts", []))
            cnn_ready = data.get("cnn_lstm") is not None
            print(
                f"[Step {step:03d}] {payload['timestamp']} | CO={payload['co_gt']:.2f} "
                f"| NO2={payload['no2_gt']:.1f} | C6H6={payload['c6h6_gt']:.1f} "
                f"| Status={status:<10} | NNs={'ON' if cnn_ready else 'WARM'} | Alerts={alert_cnt}"
            )
        except Exception as exc:
            print(f"[Step {step:03d}] Ingestion error: {exc}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
