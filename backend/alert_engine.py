THRESHOLDS = {
    "co_gt": {"elevated": 2.5, "warning": 4.0, "critical": 6.0, "unit": "mg/m3"},
    "no2_gt": {"elevated": 100.0, "warning": 150.0, "critical": 200.0, "unit": "ug/m3"},
    "c6h6_gt": {"elevated": 10.0, "warning": 15.0, "critical": 25.0, "unit": "ug/m3"},
}


def evaluate_alerts(
    actual: dict[str, float],
    xgb_pred: dict[str, float],
    is_anomaly: bool,
) -> tuple[str, list[dict[str, str]]]:
    """
    Evaluate current and forecasted gas levels.
    Returns (overall_status, list_of_alert_dicts).
    """
    alerts = []
    severity_rank = {"SAFE": 0, "ELEVATED": 1, "WARNING": 2, "CRITICAL": 3}
    overall = "SAFE"

    def bump(level: str) -> None:
        nonlocal overall
        if severity_rank[level] > severity_rank[overall]:
            overall = level

    for gas, cfg in THRESHOLDS.items():
        curr_val = float(actual[gas])
        pred_1h = float(xgb_pred[f"{gas.replace('_gt', '')}_pred_1h"])
        pred_3h = float(xgb_pred[f"{gas.replace('_gt', '')}_pred_3h"])
        unit = cfg["unit"]

        if curr_val >= cfg["critical"]:
            bump("CRITICAL")
            alerts.append({
                "severity": "CRITICAL",
                "gas": gas,
                "message": f"ACTIVE BREACH: {gas} at {curr_val:.2f} {unit} (limit {cfg['critical']})",
            })
        elif pred_1h >= cfg["warning"] or pred_3h >= cfg["warning"]:
            bump("WARNING")
            alerts.append({
                "severity": "WARNING",
                "gas": gas,
                "message": f"PREDICTED BREACH: {gas} forecast 1h={pred_1h:.2f}, 3h={pred_3h:.2f} {unit}",
            })
        elif curr_val >= cfg["elevated"]:
            bump("ELEVATED")

    if is_anomaly:
        bump("WARNING")
        alerts.append({
            "severity": "WARNING",
            "gas": "MULTI_SENSOR",
            "message": "ANOMALY DETECTED: Isolation Forest flagged abnormal telemetry signature",
        })

    return overall, alerts