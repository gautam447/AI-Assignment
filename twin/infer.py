"""Score one decision with the saved fusion model, baselines, and drivers."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from twin.features import FUSION_FEATURES, window_features
from twin.train import MODEL_PATH

HORIZONS = (30, 60, 90, 120)


def load_bundle(path: Path | None = None) -> dict:
    path = Path(path) if path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"No trained artifact at {path}. Run python -m twin.train")
    return joblib.load(path)


def _row(feats: dict, columns: list[str]) -> np.ndarray:
    return np.array([[float(feats[c]) for c in columns]], dtype=np.float64)


def _positive(model, calibrator, row: np.ndarray) -> float:
    raw = model.predict_proba(row)[:, 1]
    return float(np.clip(calibrator.predict(raw)[0], 0.0, 1.0))


def score_feature_dict(bundle: dict, feats: dict) -> dict:
    """Probabilities, 2-hour trajectory, and grouped drivers for one window."""
    hyper = {
        "persistence": _positive(bundle["persistence"], bundle["calibrators"]["persistence"], _row(feats, ["cgm_last"])),
        "static_logistic": _positive(bundle["static"], bundle["calibrators"]["static_logistic"], _row(feats, bundle["static_features"])),
        "wearable_hgb": _positive(bundle["wearable"], bundle["calibrators"]["wearable_hgb"], _row(feats, bundle["wearable_features"])),
        "fusion_hgb": _positive(bundle["fusion"], bundle["calibrators"]["fusion_hgb"], _row(feats, bundle["fusion_features"])),
    }
    fusion_row = _row(feats, bundle["fusion_features"])
    p_hypo = _positive(bundle["hypo"], bundle["hypo_calibrator"], fusion_row)
    traj = []
    for reg in bundle["trajectory"]:
        traj.append(float(reg.predict(fusion_row)[0]))
    sd = [float(v) for v in bundle["trajectory_residual_sd"]]
    return {
        "p_hyper": hyper,
        "p_hypo": p_hypo,
        "alert_hyper": hyper["fusion_hgb"] >= float(bundle["threshold_hyper"]),
        "alert_hypo": p_hypo >= float(bundle["threshold_hypo"]),
        "trajectory_mg_dl": traj,
        "trajectory_low": [c - s for c, s in zip(traj, sd)],
        "trajectory_high": [c + s for c, s in zip(traj, sd)],
        "horizons_min": list(HORIZONS),
        "drivers": local_drivers(bundle, feats, hyper["fusion_hgb"]),
    }


def score_window(bundle: dict, patient: dict, sim: dict, t: int) -> dict:
    feats = window_features(patient, sim, t)
    out = score_feature_dict(bundle, feats)
    out["features"] = feats
    out["t"] = int(t)
    return out


def local_drivers(bundle: dict, feats: dict, p_full: float | None = None) -> list[dict]:
    """How much each clinical group moves risk versus the training median.

    A positive delta means this patient's values raise hyperglycemia risk
    relative to a typical training window.
    """
    if p_full is None:
        p_full = _positive(bundle["fusion"], bundle["calibrators"]["fusion_hgb"], _row(feats, FUSION_FEATURES))
    medians = bundle["medians"]
    rows = []
    for name, members in bundle["driver_groups"]:
        altered = dict(feats)
        used = [m for m in members if m in altered]
        if not used:
            continue
        for m in used:
            altered[m] = float(medians[m])
        p_alt = _positive(bundle["fusion"], bundle["calibrators"]["fusion_hgb"], _row(altered, FUSION_FEATURES))
        rows.append({"group": name, "delta_p": p_full - p_alt, "p_without_group": p_alt})
    rows.sort(key=lambda r: r["delta_p"], reverse=True)
    return rows
