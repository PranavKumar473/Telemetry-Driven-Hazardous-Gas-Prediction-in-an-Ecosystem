import random
from typing import Any


def apply_fault(row: dict[str, Any], step: int, mode: str) -> dict[str, Any] | None:
    """
    Inject synthetic telemetry faults for stress-testing:
      - normal      : untouched historical reading
      - leak        : exponential CO & NO2 leak ramp starting at step 20
      - drift       : gradual upward sensor drift on all target gases
      - packet_loss : randomly drops 15% of packets
    """
    packet = dict(row)
    packet["fault_mode"] = mode

    if mode == "packet_loss" and random.random() < 0.15:
        return None

    if mode == "leak" and step >= 20:
        ramp = min(1.0 + 0.18 * (step - 20), 4.5)
        packet["co_gt"] = round(float(packet["co_gt"]) * ramp, 3)
        packet["no2_gt"] = round(float(packet["no2_gt"]) * (1.0 + 0.08 * (step - 20)), 3)
        packet["c6h6_gt"] = round(float(packet["c6h6_gt"]) * ramp, 3)

    elif mode == "drift":
        drift_factor = 1.0 + (step * 0.006)
        packet["co_gt"] = round(float(packet["co_gt"]) * drift_factor, 3)
        packet["no2_gt"] = round(float(packet["no2_gt"]) * drift_factor, 3)

    return packet