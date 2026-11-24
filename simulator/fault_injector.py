"""Synthetic telemetry fault injector for stress-testing live gas prediction & alerting."""
import random
from typing import Any

VALID_MODES = {"normal", "leak", "drift", "packet_loss"}


def apply_fault(
    row: dict[str, Any],
    step: int,
    mode: str = "normal",
    rng: random.Random | None = None,
) -> dict[str, Any] | None:
    """
    Inject synthetic telemetry faults for stress-testing:
      - normal      : untouched historical reading
      - leak        : exponential CO, NO2 & C6H6 leak ramp starting at step 20
      - drift       : gradual upward sensor drift on all target gases
      - packet_loss : randomly drops 15% of packets
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported fault mode '{mode}'. Expected one of { sorted(VALID_MODES) }.")

    rand = rng or random
    if mode == "packet_loss" and rand.random() < 0.15:
        return None

    packet = dict(row)
    packet["fault_mode"] = mode

    if mode == "leak" and step >= 20:
        co_ramp = min(1.0 + 0.18 * (step - 20), 4.5)
        no2_ramp = min(1.0 + 0.08 * (step - 20), 3.0)
        packet["co_gt"] = round(float(packet["co_gt"]) * co_ramp, 3)
        packet["no2_gt"] = round(float(packet["no2_gt"]) * no2_ramp, 3)
        packet["c6h6_gt"] = round(float(packet["c6h6_gt"]) * co_ramp, 3)

    elif mode == "drift":
        drift_factor = min(1.0 + (step * 0.006), 2.5)
        packet["co_gt"] = round(float(packet["co_gt"]) * drift_factor, 3)
        packet["no2_gt"] = round(float(packet["no2_gt"]) * drift_factor, 3)
        packet["c6h6_gt"] = round(float(packet["c6h6_gt"]) * drift_factor, 3)

    return packet
