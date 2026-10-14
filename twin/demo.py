"""Small synthetic cohort the dashboard opens on, with a decision time per person."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from twin.ehr import generate_ehr
from twin.infer import score_window
from twin.train import ARTIFACT_DIR
from twin.wearable import simulate_patient

DEMO_EHR = ARTIFACT_DIR / "demo_ehr.csv"
DEMO_SERIES = ARTIFACT_DIR / "demo_series.npz"
DEMO_META = ARTIFACT_DIR / "demo_meta.json"


def _snap(t: int) -> int:
    snapped = t - (t % 6)
    return max(72, snapped)


def choose_anchor(sim: dict) -> tuple[int, str]:
    """Pick a clinically readable minute: before a low, or just after a meal."""
    cgm = np.asarray(sim["cgm"], dtype=float)
    n = cgm.shape[0]
    hypo_choice = None
    for t in range(72, n - 24, 6):
        future = cgm[t + 1 : t + 25]
        if float(future.min()) < 70 and float(cgm[t]) >= 75:
            depth = float(future.min())
            if hypo_choice is None or depth < hypo_choice[0]:
                hypo_choice = (depth, t)
    if hypo_choice is not None:
        return int(hypo_choice[1]), "30–120 min before glucose falls below 70 mg/dL"

    best = None
    meals = np.asarray(sim["meal_carb"])
    minute = np.asarray(sim["minute_of_day"])
    for t in range(72, n - 24, 6):
        recent = float(meals[max(0, t - 8) : t + 1].sum())
        slope = float(cgm[t] - cgm[t - 6])
        if recent >= 40 and 125 <= cgm[t] <= 240 and slope > -2:
            lunch = 12 * 60 <= int(minute[t]) <= 15 * 60 + 30
            score = slope + recent / 20.0 + (25.0 if lunch else 0.0) + max(0.0, cgm[t] - 140) / 10.0
            if best is None or score > best[0]:
                best = (score, t)
    if best is not None:
        return int(best[1]), "after a meal, while glucose is still moving"
    # Last resort: highest glucose that still has a 2-hour future.
    t = int(72 + np.argmax(cgm[72 : n - 24]))
    return _snap(t), "highest glucose in the simulated record"


def _story_candidates(sim: dict) -> list[tuple[int, str]]:
    """Decision times worth showing: after a meal, before a meal, or before a low."""
    cgm = np.asarray(sim["cgm"], dtype=float)
    meal = np.asarray(sim["meal_carb"], dtype=float)
    n = cgm.shape[0]
    found = []
    for t in range(72, n - 24, 6):
        recent = float(meal[max(0, t - 12) : t + 1].sum())
        planned = float(meal[t + 1 : t + 19].sum())
        future_min = float(cgm[t + 1 : t + 25].min())
        g = float(cgm[t])
        if future_min < 70.0 and g >= 80.0:
            found.append((t, "before a simulated low"))
        elif 115.0 <= g <= 230.0 and recent >= 30.0:
            found.append((t, "after a meal, while glucose is still moving"))
        elif 100.0 <= g <= 190.0 and planned >= 40.0 and recent < 20.0:
            found.append((t, "before a logged meal"))
    return found


def build_demo(bundle: dict, n: int = 12, days: int = 3, seed: int = 11, pool: int = 36) -> dict:
    """Pick a 12-person panel that spans low, uncertain, and high risk, plus lows.

    Post-meal windows in this simulator are often near-certain. The panel
    deliberately keeps mid-range probabilities so a clinician can see the
    model hesitate, not only shout.
    """
    ehr = generate_ehr(pool, seed=seed)
    library = []
    for _, row in ehr.iterrows():
        patient = row.to_dict()
        sim = simulate_patient(patient, days=days, enable_random_interventions=False)
        options = []
        for t, reason in _story_candidates(sim):
            scored = score_window(bundle, patient, sim, t)
            options.append((t, reason, scored))
        if not options:
            t, reason = choose_anchor(sim)
            options.append((t, reason, score_window(bundle, patient, sim, t)))
        library.append((patient, sim, options))

    used: set[int] = set()
    chosen = []

    hypo_hits = []
    for i, (_patient, _sim, options) in enumerate(library):
        for t, reason, scored in options:
            if scored["p_hypo"] >= 0.45 and scored["p_hyper"]["fusion_hgb"] <= 0.35:
                hypo_hits.append((scored["p_hypo"], i, t, reason, scored))
    hypo_hits.sort(reverse=True)
    for _p, i, t, reason, scored in hypo_hits:
        if i in used:
            continue
        used.add(i)
        patient, sim, _options = library[i]
        chosen.append((patient, sim, t, reason, scored))
        if sum(1 for c in chosen if c[3] == "before a simulated low") >= 3:
            break

    # Targets across the probability range, filled from patients not yet used.
    for target in (0.96, 0.84, 0.68, 0.52, 0.38, 0.24, 0.12):
        best = None
        for i, (patient, sim, options) in enumerate(library):
            if i in used:
                continue
            for t, reason, scored in options:
                p = scored["p_hyper"]["fusion_hgb"]
                if reason == "before a simulated low":
                    continue
                dist = abs(p - target)
                if best is None or dist < best[0]:
                    best = (dist, i, t, reason, scored)
        if best is None or best[0] > 0.2:
            continue
        _dist, i, t, reason, scored = best
        used.add(i)
        patient, sim, _options = library[i]
        chosen.append((patient, sim, t, reason, scored))

    if len(chosen) < n:
        rest = []
        for i, (patient, sim, options) in enumerate(library):
            if i in used or not options:
                continue
            t, reason, scored = max(options, key=lambda item: item[2]["p_hyper"]["fusion_hgb"])
            rest.append((scored["p_hyper"]["fusion_hgb"], i, t, reason, scored))
        rest.sort(reverse=True)
        for _p, i, t, reason, scored in rest:
            if len(chosen) >= n:
                break
            patient, sim, _options = library[i]
            chosen.append((patient, sim, t, reason, scored))
    chosen = chosen[:n]
    chosen.sort(key=lambda item: item[4]["p_hyper"]["fusion_hgb"], reverse=True)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ehr_out = pd.DataFrame([p[0] for p in chosen])
    ehr_out.to_csv(DEMO_EHR, index=False)
    series = {
        "cgm": np.stack([p[1]["cgm"] for p in chosen]),
        "hr": np.stack([p[1]["hr"] for p in chosen]),
        "hrv_rmssd": np.stack([p[1]["hrv_rmssd"] for p in chosen]),
        "steps": np.stack([p[1]["steps"] for p in chosen]),
        "sleep_stage": np.stack([p[1]["sleep_stage"] for p in chosen]),
        "minute_of_day": np.stack([p[1]["minute_of_day"] for p in chosen]),
        "day_index": np.stack([p[1]["day_index"] for p in chosen]),
        "meal_carb": np.stack([p[1]["meal_carb"] for p in chosen]),
        "insulin_units": np.stack([p[1]["insulin_units"] for p in chosen]),
        "su_dose": np.stack([p[1]["su_dose"] for p in chosen]),
        "walk_now": np.stack([p[1]["walk_now"] for p in chosen]),
        "skip_next_meal": np.stack([p[1]["skip_next_meal"] for p in chosen]),
        "anchor": np.asarray([p[2] for p in chosen], dtype=np.int32),
    }
    np.savez_compressed(DEMO_SERIES, **series)
    meta = {
        "seed": seed,
        "days": days,
        "pool": pool,
        "patients": [
            {
                "patient_id": p[0]["patient_id"],
                "anchor": int(p[2]),
                "anchor_reason": p[3],
                "p_hyper_fusion": p[4]["p_hyper"]["fusion_hgb"],
                "p_hypo": p[4]["p_hypo"],
                "alert_hyper": bool(p[4]["alert_hyper"]),
                "alert_hypo": bool(p[4]["alert_hypo"]),
            }
            for p in chosen
        ],
    }
    DEMO_META.write_text(json.dumps(meta, indent=2))
    return meta


def load_demo() -> tuple[pd.DataFrame, dict]:
    if not DEMO_EHR.exists() or not DEMO_SERIES.exists():
        raise FileNotFoundError("Demo cohort is missing. Run python run_demo.py")
    ehr = pd.read_csv(DEMO_EHR)
    series = np.load(DEMO_SERIES)
    return ehr, {key: series[key] for key in series.files}


def sim_from_row(series: dict, index: int) -> dict:
    return {key: series[key][index] for key in series if key != "anchor"}
