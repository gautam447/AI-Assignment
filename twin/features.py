"""Decision-time features from a 6-hour window plus the static chart.

Labels look only at the next 2 hours. Meal and dose fields are the log a
patient app would already have at the decision minute; they are not sensor
aggregates, and the wearable baseline does not receive them.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from twin.wearable import DT_MIN, HYPER_MGDL, HYPO_MGDL

WINDOW_BINS = 72  # 6 hours at 5 min
HORIZON_BINS = 24  # 2 hours
STRIDE_BINS = 6  # score every 30 min

STATIC_FEATURES = [
    "age",
    "sex_male",
    "bmi",
    "diabetes_years",
    "hba1c_pct",
    "fasting_glucose_mg_dl",
    "ldl_mg_dl",
    "hdl_mg_dl",
    "triglycerides_mg_dl",
    "egfr_ml_min",
    "sbp_mmhg",
    "family_history_t2d",
    "tcf7l2_alleles",
    "cdkal1_alleles",
    "on_metformin",
    "on_sulfonylurea",
    "insulin_total_daily_units",
    "diet_rice_forward",
    "urban",
]

WEARABLE_FEATURES = [
    "cgm_last",
    "cgm_mean",
    "cgm_std",
    "cgm_min",
    "cgm_max",
    "cgm_slope_1h",
    "cgm_slope_3h",
    "cgm_accel",
    "frac_above_140",
    "frac_above_180",
    "frac_below_70",
    "hr_mean",
    "hr_last",
    "hrv_mean",
    "steps_6h",
    "steps_30m",
    "sleep_frac",
    "deep_frac",
    "hour_sin",
    "hour_cos",
]

CONTEXT_FEATURES = [
    "minutes_since_meal",
    "carbs_last_2h",
    "planned_carbs_next_2h",
    "minutes_since_insulin",
    "insulin_units_recent",
    "minutes_since_su",
    "walk_now",
    "skip_next_meal",
]

FUSION_FEATURES = STATIC_FEATURES + WEARABLE_FEATURES + CONTEXT_FEATURES

DRIVER_GROUPS = [
    ("Recent glucose level", ["cgm_last", "cgm_mean", "cgm_min", "cgm_max", "frac_above_140", "frac_above_180", "frac_below_70"]),
    ("Glucose slope and variability", ["cgm_std", "cgm_slope_1h", "cgm_slope_3h", "cgm_accel"]),
    ("Heart rate, HRV, steps, sleep", ["hr_mean", "hr_last", "hrv_mean", "steps_6h", "steps_30m", "sleep_frac", "deep_frac"]),
    ("Time of day", ["hour_sin", "hour_cos"]),
    ("HbA1c, body size, diabetes duration", ["hba1c_pct", "fasting_glucose_mg_dl", "bmi", "diabetes_years", "age"]),
    ("Lipids, kidney, blood pressure", ["ldl_mg_dl", "hdl_mg_dl", "triglycerides_mg_dl", "egfr_ml_min", "sbp_mmhg"]),
    ("Medications on the chart", ["on_metformin", "on_sulfonylurea", "insulin_total_daily_units"]),
    ("Family history and genetic markers", ["family_history_t2d", "tcf7l2_alleles", "cdkal1_alleles", "sex_male", "diet_rice_forward", "urban"]),
    ("Meal and insulin log", ["minutes_since_meal", "carbs_last_2h", "planned_carbs_next_2h", "minutes_since_insulin", "insulin_units_recent", "minutes_since_su"]),
]


def _slope_per_hour(y: np.ndarray) -> float:
    n = int(y.shape[0])
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64) * (DT_MIN / 60.0)
    x -= x.mean()
    yc = y.astype(np.float64) - float(y.mean())
    denom = float(np.dot(x, x))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(x, yc) / denom)


def _minutes_since(events: np.ndarray, t: int, default: float) -> float:
    prior = np.flatnonzero(events[: t + 1] > 0)
    if prior.size == 0:
        return default
    return float((t - int(prior[-1])) * DT_MIN)


def decision_indices(n: int) -> range:
    return range(WINDOW_BINS, n - HORIZON_BINS, STRIDE_BINS)


def window_features(patient: Mapping, sim: Mapping, t: int) -> dict:
    """Features known at bin ``t`` (inclusive). Does not read glucose after ``t``."""
    if t < WINDOW_BINS - 1:
        raise ValueError("not enough history for a 6-hour window")
    cgm = sim["cgm"]
    start = t - WINDOW_BINS + 1
    g = np.asarray(cgm[start : t + 1], dtype=np.float64)
    hr = np.asarray(sim["hr"][start : t + 1], dtype=np.float64)
    hrv = np.asarray(sim["hrv_rmssd"][start : t + 1], dtype=np.float64)
    steps = np.asarray(sim["steps"][start : t + 1], dtype=np.float64)
    sleep = np.asarray(sim["sleep_stage"][start : t + 1])
    slope_1h = _slope_per_hour(g[-12:])
    slope_prev = _slope_per_hour(g[-24:-12]) if g.shape[0] >= 24 else slope_1h
    minute = int(sim["minute_of_day"][t])
    angle = 2.0 * np.pi * (minute / (24.0 * 60.0))
    meal = np.asarray(sim["meal_carb"])
    insulin = np.asarray(sim["insulin_units"])
    su = np.asarray(sim["su_dose"])
    mins_ins = _minutes_since(insulin, t, default=24 * 60)
    last_ins = np.flatnonzero(insulin[: t + 1] > 0)
    ins_units = float(insulin[int(last_ins[-1])]) if last_ins.size and mins_ins <= 360 else 0.0
    feats = {
        "cgm_last": float(g[-1]),
        "cgm_mean": float(g.mean()),
        "cgm_std": float(g.std()),
        "cgm_min": float(g.min()),
        "cgm_max": float(g.max()),
        "cgm_slope_1h": slope_1h,
        "cgm_slope_3h": _slope_per_hour(g[-36:]) if g.shape[0] >= 36 else slope_1h,
        "cgm_accel": slope_1h - slope_prev,
        "frac_above_140": float(np.mean(g >= 140.0)),
        "frac_above_180": float(np.mean(g >= HYPER_MGDL)),
        "frac_below_70": float(np.mean(g < HYPO_MGDL)),
        "hr_mean": float(hr.mean()),
        "hr_last": float(hr[-1]),
        "hrv_mean": float(hrv.mean()),
        "steps_6h": float(steps.sum()),
        "steps_30m": float(steps[-6:].sum()),
        "sleep_frac": float(np.mean(sleep > 0)),
        "deep_frac": float(np.mean(sleep == 2)),
        "hour_sin": float(np.sin(angle)),
        "hour_cos": float(np.cos(angle)),
        "minutes_since_meal": _minutes_since(meal, t, default=12 * 60),
        "carbs_last_2h": float(meal[max(0, t - 23) : t + 1].sum()),
        "planned_carbs_next_2h": float(meal[t + 1 : t + 1 + HORIZON_BINS].sum()) if t + 1 < meal.shape[0] else 0.0,
        "minutes_since_insulin": mins_ins,
        "insulin_units_recent": ins_units,
        "minutes_since_su": _minutes_since(su, t, default=24 * 60),
        "walk_now": float(sim["walk_now"][t]),
        "skip_next_meal": float(sim["skip_next_meal"][t]),
    }
    for name in STATIC_FEATURES:
        feats[name] = float(patient[name])
    return feats


def window_labels(sim: Mapping, t: int) -> dict:
    future = np.asarray(sim["cgm"][t + 1 : t + 1 + HORIZON_BINS], dtype=np.float64)
    if future.shape[0] < HORIZON_BINS:
        raise ValueError("not enough future for a 2-hour label")
    return {
        "y_hyper": int(np.max(future) >= HYPER_MGDL),
        "y_hypo": int(np.min(future) < HYPO_MGDL),
        "y30": float(future[5]),
        "y60": float(future[11]),
        "y90": float(future[17]),
        "y120": float(future[23]),
    }


def extract_patient_table(patient: Mapping, sim: Mapping) -> dict[str, np.ndarray]:
    """All 30-minute decisions for one simulated twin."""
    n = int(np.asarray(sim["cgm"]).shape[0])
    times = list(decision_indices(n))
    rows = []
    labels = []
    for t in times:
        rows.append(window_features(patient, sim, t))
        labels.append(window_labels(sim, t))
    if not rows:
        raise RuntimeError("simulation produced no decision windows")
    out: dict[str, np.ndarray] = {
        "patient_id": np.array([patient["patient_id"]] * len(rows)),
        "t": np.asarray(times, dtype=np.int32),
        "day_index": np.asarray([int(sim["day_index"][t]) for t in times], dtype=np.int16),
    }
    for name in FUSION_FEATURES:
        out[name] = np.asarray([r[name] for r in rows], dtype=np.float64)
    for name in ("y_hyper", "y_hypo", "y30", "y60", "y90", "y120"):
        out[name] = np.asarray([r[name] for r in labels], dtype=np.float64)
    return out


def stack_tables(tables: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = tables[0].keys()
    stacked = {}
    for key in keys:
        stacked[key] = np.concatenate([t[key] for t in tables], axis=0)
    return stacked


def matrix(table: dict[str, np.ndarray], columns: list[str]) -> np.ndarray:
    return np.column_stack([table[c] for c in columns])
