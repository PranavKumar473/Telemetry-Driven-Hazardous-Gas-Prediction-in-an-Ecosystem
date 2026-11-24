import argparse
import time
from pathlib import Path
import pandas as pd
import requests
from simulator.fault_injector import apply_fault

STREAM_PATH = Path("data/processed/test_stream.csv")
API_URL = "http://127.0.0.1:8000/ingest"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Telemetry Simulator")
    parser.add_argument("--mode", choices=["normal", "leak", "drift", "packet_loss"], default="normal")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between packets")
    parser.add_argument("--limit", type=int, default=150, help="Max packets to replay")
    parser.add_argument("--node", type=str, default="eco_node_01")
    args = parser.parse_args()

    df = pd.read_csv(STREAM_PATH).head(args.limit)
    print(f"Streaming {len(df)} packets | node={args.node} | mode={args.mode} | interval={args.interval}s")

    for step, (_, row) in enumerate(df.iterrows(), start=1):
        payload = row.to_dict()
        payload["node_id"] = args.node
        payload = apply_fault(payload, step, args.mode)

        if payload is None:
            print(f"[Step {step:03d}] DROPPED PACKET (simulated packet_loss)")
            time.sleep(args.interval)
            continue

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            data = resp.json()
            status = data.get("status", "UNKNOWN")
            alert_cnt = len(data.get("alerts", []))
            print(
                f"[Step {step:03d}] {payload['timestamp']} | CO={payload['co_gt']:.2f} "
                f"| NO2={payload['no2_gt']:.1f} | Status={status} | Alerts={alert_cnt}"
            )
        except Exception as exc:
            print(f"[Step {step:03d}] Ingestion error: {exc}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()