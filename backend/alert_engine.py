"""Stateful tiered alerting engine with cooldown and hysteresis.

Separates Hazardous Gas Breach Forecasts (category='GAS_BREACH') from
Sensor Hardware / Anomaly Faults (category='SENSOR_ANOMALY') and prevents
duplicate alert spam across consecutive packets.
"""
from dataclasses import dataclass
from typing import Any

THRESHOLDS: dict[str, dict[str, Any]] = {
    "co_gt": {"short": "co", "elevated": 2.5, "warning": 4.0, "critical": 6.0, "unit": "mg/m3"},
    "no2_gt": {"short": "no2", "elevated": 100.0, "warning": 150.0, "critical": 200.0, "unit": "ug/m3"},
    "c6h6_gt": {"short": "c6h6", "elevated": 10.0, "warning": 15.0, "critical": 25.0, "unit": "ug/m3"},
}

SEVERITY_RANK = {"SAFE": 0, "ELEVATED": 1, "WARNING": 2, "CRITICAL": 3}
DEFAULT_COOLDOWN_STEPS = 5
HYSTERESIS_RATIO = 0.90


@dataclass
class _ChannelState:
    severity: str = "SAFE"
    steps_since_emit: int = 0


class AlertCooldownManager:
    """
    Per-(node_id, category, gas) state tracker that enforces:
      1. Immediate emission when severity escalates (e.g. SAFE -> WARNING -> CRITICAL).
      2. Cooldown suppression for `cooldown_steps` packets when severity stays the same.
      3. Hysteresis clearing: a channel only drops to a lower tier once values fall
         below `hysteresis_ratio` (90%) of the active tier's threshold.
    """

    def __init__(self, cooldown_steps: int = DEFAULT_COOLDOWN_STEPS, hysteresis_ratio: float = HYSTERESIS_RATIO) -> None:
        self.cooldown_steps = cooldown_steps
        self.hysteresis_ratio = hysteresis_ratio
        self._state: dict[tuple[str, str, str], _ChannelState] = {}

    def reset(self, node_id: str | None = None) -> None:
        if node_id is None:
            self._state.clear()
        else:
            for key in [k for k in self._state if k[0] == node_id]:
                del self._state[key]

    def _should_emit(self, key: tuple[str, str, str], candidate_severity: str, clear_below_hysteresis: bool) -> str:
        st = self._state.setdefault(key, _ChannelState())
        cand_rank = SEVERITY_RANK[candidate_severity]
        prev_rank = SEVERITY_RANK[st.severity]

        if cand_rank == 0:
            if prev_rank > 0 and not clear_below_hysteresis:
                # Hold previous severity inside hysteresis band without re-emitting
                st.steps_since_emit += 1
                return "SUPPRESSED"
            st.severity = "SAFE"
            st.steps_since_emit = 0
            return "SAFE"

        # Escalation: always emit immediately
        if cand_rank > prev_rank:
            st.severity = candidate_severity
            st.steps_since_emit = 0
            return candidate_severity

        # Same or de-escalating non-safe tier: apply cooldown
        st.steps_since_emit += 1
        st.severity = candidate_severity
        if st.steps_since_emit >= self.cooldown_steps:
            st.steps_since_emit = 0
            return candidate_severity
        return "SUPPRESSED"

    def evaluate(
        self,
        actual: dict[str, Any],
        xgb_pred: dict[str, float],
        is_anomaly: bool,
        lstm_pred: dict[str, float] | None = None,
        cnn_lstm_pred: dict[str, float] | None = None,
    ) -> tuple[str, list[dict[str, str]]]:
        node_id = str(actual.get("node_id", "eco_node_01"))
        fault_mode = str(actual.get("fault_mode", "normal"))
        alerts: list[dict[str, str]] = []
        overall = "SAFE"

        def bump(level: str) -> None:
            nonlocal overall
            if SEVERITY_RANK[level] > SEVERITY_RANK[overall]:
                overall = level

        model_forecasts = [m for m in (xgb_pred, lstm_pred, cnn_lstm_pred) if m is not None]

        # --- 1. Hazardous Gas Breach Evaluation (category='GAS_BREACH') ---
        for gas, cfg in THRESHOLDS.items():
            short = cfg["short"]
            unit = cfg["unit"]
            curr_val = float(actual[gas])

            peak_1h = max(float(m[f"{short}_pred_1h"]) for m in model_forecasts)
            peak_3h = max(float(m[f"{short}_pred_3h"]) for m in model_forecasts)
            peak_6h = max(float(m[f"{short}_pred_6h"]) for m in model_forecasts)
            peak_forecast = max(peak_1h, peak_3h, peak_6h)

            cand_sev = "SAFE"
            msg = ""
            active_thresh = cfg["elevated"]

            if curr_val >= cfg["critical"]:
                cand_sev = "CRITICAL"
                active_thresh = cfg["critical"]
                msg = f"ACTIVE CRITICAL BREACH: {gas} at {curr_val:.2f} {unit} (critical limit {cfg['critical']} {unit})"
            elif max(peak_1h, peak_3h) >= cfg["critical"]:
                cand_sev = "CRITICAL"
                active_thresh = cfg["critical"]
                msg = (
                    f"FORECAST CRITICAL BREACH: {gas} projected to reach "
                    f"1h={peak_1h:.2f}, 3h={peak_3h:.2f} {unit} (critical limit {cfg['critical']} {unit})"
                )
            elif curr_val >= cfg["warning"] or peak_forecast >= cfg["warning"]:
                cand_sev = "WARNING"
                active_thresh = cfg["warning"]
                msg = (
                    f"PREDICTED BREACH: {gas} current={curr_val:.2f}, "
                    f"forecast 1h={peak_1h:.2f}, 3h={peak_3h:.2f}, 6h={peak_6h:.2f} {unit} "
                    f"(warning limit {cfg['warning']} {unit})"
                )
            elif curr_val >= cfg["elevated"]:
                cand_sev = "ELEVATED"
                active_thresh = cfg["elevated"]
                msg = f"ELEVATED CONCENTRATION: {gas} at {curr_val:.2f} {unit} (elevated threshold {cfg['elevated']} {unit})"

            key = (node_id, "GAS_BREACH", gas)
            prev_sev = self._state.get(key, _ChannelState()).severity
            prev_thresh = cfg.get(prev_sev.lower(), cfg["elevated"])
            clear_ok = max(curr_val, peak_forecast) < (prev_thresh * self.hysteresis_ratio)

            decision = self._should_emit(key, cand_sev, clear_below_hysteresis=clear_ok)
            effective_sev = self._state[key].severity
            bump(effective_sev)

            if decision in ("WARNING", "CRITICAL"):
                alerts.append({
                    "severity": decision,
                    "category": "GAS_BREACH",
                    "gas": gas,
                    "message": msg,
                })

        # --- 2. Sensor Hardware / Anomaly Fault Evaluation (category='SENSOR_ANOMALY') ---
        anom_key = (node_id, "SENSOR_ANOMALY", "MULTI_SENSOR")
        if is_anomaly:
            decision = self._should_emit(anom_key, "WARNING", clear_below_hysteresis=False)
            bump("WARNING")
            if decision == "WARNING":
                alerts.append({
                    "severity": "WARNING",
                    "category": "SENSOR_ANOMALY",
                    "gas": "MULTI_SENSOR",
                    "message": (
                        f"SENSOR ANOMALY DETECTED: Isolation Forest flagged out-of-distribution "
                        f"multivariate telemetry signature (fault_mode={fault_mode})"
                    ),
                })
        else:
            self._should_emit(anom_key, "SAFE", clear_below_hysteresis=True)

        return overall, alerts


DEFAULT_ALERT_MANAGER = AlertCooldownManager()


def evaluate_alerts(
    actual: dict[str, Any],
    xgb_pred: dict[str, float],
    is_anomaly: bool,
    lstm_pred: dict[str, float] | None = None,
    cnn_lstm_pred: dict[str, float] | None = None,
    manager: AlertCooldownManager | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Evaluate alerts using the shared stateful cooldown manager (or a custom one)."""
    active_manager = manager or DEFAULT_ALERT_MANAGER
    return active_manager.evaluate(
        actual=actual,
        xgb_pred=xgb_pred,
        is_anomaly=is_anomaly,
        lstm_pred=lstm_pred,
        cnn_lstm_pred=cnn_lstm_pred,
    )
