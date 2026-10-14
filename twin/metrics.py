"""Ranking, calibration, and alert-usefulness metrics for the 2-hour task."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def operating_point(y: np.ndarray, p: np.ndarray, target_spec: float = 0.80) -> dict:
    """Sensitivity at the most sensitive threshold whose specificity is at least ``target_spec``."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return {"sensitivity": float("nan"), "specificity": float("nan"), "threshold": 1.0}
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1.0 - fpr
    valid = np.flatnonzero(spec >= target_spec - 1e-12)
    if valid.size == 0:
        i = int(np.argmax(spec))
    else:
        best_tpr = tpr[valid].max()
        cand = valid[np.flatnonzero(np.isclose(tpr[valid], best_tpr))]
        i = int(cand[np.argmin(spec[cand])])
    threshold = float(thr[i]) if np.isfinite(thr[i]) else 1.0
    return {
        "sensitivity": float(tpr[i]),
        "specificity": float(spec[i]),
        "threshold": threshold,
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(1, y.shape[0])
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(p[mask].mean())
        rate = float(y[mask].mean())
        ece += (count / total) * abs(rate - confidence)
        rows.append(
            {
                "bin_left": float(lo),
                "bin_right": float(hi),
                "count": count,
                "mean_predicted": round(confidence, 4),
                "observed_rate": round(rate, 4),
            }
        )
    return float(ece), rows


def classification_report(y: np.ndarray, p: np.ndarray, target_spec: float = 0.80) -> dict:
    point = operating_point(y, p, target_spec=target_spec)
    ece, curve = expected_calibration_error(y, p)
    return {
        "auroc": safe_auroc(y, p),
        "auprc": safe_auprc(y, p),
        "sensitivity_at_specificity": point["sensitivity"],
        "achieved_specificity": point["specificity"],
        "threshold": point["threshold"],
        "ece": ece,
        "calibration_bins": curve,
        "n": int(np.asarray(y).shape[0]),
        "prevalence": float(np.mean(y)),
    }


def lead_time_report(
    cgm: np.ndarray,
    decision_t: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    hyper_mgdl: float = 180.0,
    horizon_bins: int = 24,
) -> dict:
    """Lead time for hyperglycemia episodes on one patient timeline.

    An alert counts only when glucose is still under 180 mg/dL. Lead time is
    the gap from the first such alert in the preceding 2 hours to the moment
    CGM reaches 180.
    """
    cgm = np.asarray(cgm, dtype=float)
    decision_t = np.asarray(decision_t, dtype=int)
    proba = np.asarray(proba, dtype=float)
    onsets = []
    above = cgm >= hyper_mgdl
    for i in range(1, cgm.shape[0]):
        if above[i] and not above[i - 1]:
            if not onsets or i - onsets[-1] > 6:
                onsets.append(i)
    detected = []
    missed = 0
    usable = 0
    for onset in onsets:
        window = [
            (int(t), float(p))
            for t, p in zip(decision_t, proba)
            if onset - horizon_bins <= int(t) < onset and cgm[int(t)] < hyper_mgdl
        ]
        if not window:
            continue
        usable += 1
        hits = [(t, p) for t, p in window if p >= threshold]
        if not hits:
            missed += 1
            continue
        first = min(hits, key=lambda pair: pair[0])
        detected.append((onset - first[0]) * 5)
    alert_on = (proba >= threshold) & (cgm[decision_t] < hyper_mgdl)
    alerts = int(np.sum(alert_on))
    starts = 0
    prev = False
    for flag in alert_on:
        if flag and not prev:
            starts += 1
        prev = bool(flag)
    true_alerts = 0
    for t, p in zip(decision_t, proba):
        if p < threshold or cgm[int(t)] >= hyper_mgdl:
            continue
        future = cgm[int(t) + 1 : int(t) + 1 + horizon_bins]
        if future.size and np.max(future) >= hyper_mgdl:
            true_alerts += 1
    return {
        "usable_episodes": usable,
        "detected": len(detected),
        "missed": missed,
        "lead_minutes": detected,
        "alerts_while_below_180": alerts,
        "alert_starts": starts,
        "true_alerts": true_alerts,
    }


def summarize_lead(parts: list[dict], patient_days: float) -> dict:
    leads = [m for part in parts for m in part["lead_minutes"]]
    usable = sum(p["usable_episodes"] for p in parts)
    detected = sum(p["detected"] for p in parts)
    alerts = sum(p["alerts_while_below_180"] for p in parts)
    starts = sum(p["alert_starts"] for p in parts)
    true_alerts = sum(p["true_alerts"] for p in parts)
    leads_arr = np.asarray(leads, dtype=float) if leads else np.asarray([])
    return {
        "usable_episodes": int(usable),
        "detection_rate": float(detected / usable) if usable else float("nan"),
        "median_lead_min": float(np.median(leads_arr)) if leads_arr.size else float("nan"),
        "pct_lead_at_least_30_min": float(np.mean(leads_arr >= 30)) if leads_arr.size else float("nan"),
        "alerts_per_patient_day": float(alerts / patient_days) if patient_days else float("nan"),
        "alert_episodes_per_patient_day": float(starts / patient_days) if patient_days else float("nan"),
        "alert_ppv": float(true_alerts / alerts) if alerts else float("nan"),
        "n_detected": int(detected),
    }
