"""Train baselines and the fusion model, then write artifacts/metrics.json.

Patient-level split: no window from a test patient is seen in training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from twin.ehr import generate_ehr
from twin.features import (
    CONTEXT_FEATURES,
    DRIVER_GROUPS,
    FUSION_FEATURES,
    STATIC_FEATURES,
    WEARABLE_FEATURES,
    extract_patient_table,
    matrix,
    stack_tables,
    window_features,
)
from twin.metrics import classification_report, lead_time_report, summarize_lead
from twin.wearable import HYPER_MGDL, simulate_patient

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

TARGET_SPEC = 0.80


def _hgb_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_depth=6,
        min_samples_leaf=40,
        l2_regularization=0.15,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=15,
        random_state=seed,
    )


def _hgb_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=200,
        max_depth=6,
        min_samples_leaf=40,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=12,
        random_state=seed,
    )


def build_dataset(n_patients: int, days: int, seed: int) -> tuple[dict[str, np.ndarray], object]:
    ehr = generate_ehr(n_patients, seed=seed)
    tables = []
    for _, row in ehr.iterrows():
        patient = row.to_dict()
        sim = simulate_patient(patient, days=days, enable_random_interventions=True)
        tables.append(extract_patient_table(patient, sim))
    return stack_tables(tables), ehr


def _split_ids(patient_ids: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    uniq = np.array(sorted(set(patient_ids.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = uniq.shape[0]
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    return {
        "train": uniq[:n_train],
        "val": uniq[n_train : n_train + n_val],
        "test": uniq[n_train + n_val :],
    }


def _mask(table, ids) -> np.ndarray:
    id_set = set(ids.tolist())
    return np.array([pid in id_set for pid in table["patient_id"]], dtype=bool)


def _fit_iso(raw: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, y.astype(float))
    return iso


def _positive(model, calibrator, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)[:, 1]
    return np.asarray(calibrator.predict(raw), dtype=float)


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:
            return None
        return round(obj, 4)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if value != value:
            return None
        return round(value, 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _group_permutation(model, calibrator, x: np.ndarray, y: np.ndarray, columns: list[str], seed: int) -> list[dict]:
    from twin.metrics import safe_auroc

    base = safe_auroc(y, _positive(model, calibrator, x))
    rng = np.random.default_rng(seed)
    rows = []
    # One shared permutation keeps the comparison about the feature group.
    order = rng.permutation(x.shape[0])
    for name, members in DRIVER_GROUPS:
        idx = [columns.index(m) for m in members if m in columns]
        if not idx:
            continue
        shuffled = x.copy()
        shuffled[:, idx] = shuffled[order][:, idx]
        score = safe_auroc(y, _positive(model, calibrator, shuffled))
        rows.append({"group": name, "auroc_drop": base - score})
    rows.sort(key=lambda r: r["auroc_drop"], reverse=True)
    return rows


def _action_shift(model, calibrator, x: np.ndarray, columns: list[str]) -> dict:
    """Mean probability change when a logged action is forced on, on near-threshold windows."""
    last = x[:, columns.index("cgm_last")]
    slope = x[:, columns.index("cgm_slope_1h")]
    mask = (last >= 140) & (last < 190) & (slope > 0)
    if int(mask.sum()) < 30:
        mask = (last >= 130) & (last < 200)
    subset = x[mask]
    if subset.shape[0] == 0:
        return {"n": 0, "walk_delta": None, "skip_delta": None}
    base = _positive(model, calibrator, subset).mean()
    walked = subset.copy()
    walked[:, columns.index("walk_now")] = 1.0
    skipped = subset.copy()
    skipped[:, columns.index("skip_next_meal")] = 1.0
    skipped[:, columns.index("planned_carbs_next_2h")] = 0.0
    return {
        "n": int(subset.shape[0]),
        "mean_p_hyper": float(base),
        "walk_delta": float(_positive(model, calibrator, walked).mean() - base),
        "skip_delta": float(_positive(model, calibrator, skipped).mean() - base),
    }


def train(n_patients: int = 320, days: int = 3, seed: int = 2026, save: bool = True) -> dict:
    print(f"Generating {n_patients} synthetic patients x {days} days (seed {seed})...")
    table, ehr = build_dataset(n_patients, days, seed)
    splits = _split_ids(table["patient_id"], seed)
    masks = {name: _mask(table, ids) for name, ids in splits.items()}
    y = table["y_hyper"].astype(int)
    y_hypo = table["y_hypo"].astype(int)
    print(
        "Windows",
        {name: int(mask.sum()) for name, mask in masks.items()},
        "hyper prevalence",
        {name: round(float(y[mask].mean()), 3) for name, mask in masks.items()},
    )

    x_static = matrix(table, STATIC_FEATURES)
    x_wear = matrix(table, WEARABLE_FEATURES)
    x_fusion = matrix(table, FUSION_FEATURES)
    x_persist = table["cgm_last"].reshape(-1, 1)

    print("Fitting persistence, static logistic, wearable HGB, fusion HGB...")
    persistence = LogisticRegression(max_iter=400, solver="lbfgs", random_state=seed)
    persistence.fit(x_persist[masks["train"]], y[masks["train"]])

    static = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs", C=0.5, random_state=seed)),
        ]
    )
    static.fit(x_static[masks["train"]], y[masks["train"]])

    wearable = _hgb_classifier(seed)
    wearable.fit(x_wear[masks["train"]], y[masks["train"]])

    # Ordinary windows outnumber logged walks and skipped meals. Upweight
    # those rows so the counterfactual flags are actually fit, then say so
    # in the model card. The test set is not reweighted.
    action = ((table["walk_now"] == 1) | (table["skip_next_meal"] == 1)).astype(float)
    fusion_weight = np.ones(masks["train"].sum(), dtype=float)
    fusion_weight[action[masks["train"]] == 1] = 3.0
    fusion = _hgb_classifier(seed + 1)
    fusion.fit(x_fusion[masks["train"]], y[masks["train"]], sample_weight=fusion_weight)

    hypo = _hgb_classifier(seed + 2)
    hypo.fit(x_fusion[masks["train"]], y_hypo[masks["train"]], sample_weight=fusion_weight)

    print("Fitting 30/60/90/120 min trajectory models...")
    traj_models = []
    residual_sd = []
    traj_mae = {}
    coverage = {}
    horizons = [("30", "y30"), ("60", "y60"), ("90", "y90"), ("120", "y120")]
    for offset, (label, key) in enumerate(horizons):
        reg = _hgb_regressor(seed + 10 + offset)
        target = table[key]
        reg.fit(x_fusion[masks["train"]], target[masks["train"]])
        val_pred = reg.predict(x_fusion[masks["val"]])
        resid = target[masks["val"]] - val_pred
        sd = float(max(6.0, np.std(resid)))
        test_pred = reg.predict(x_fusion[masks["test"]])
        err = np.abs(target[masks["test"]] - test_pred)
        band = np.abs(target[masks["test"]] - test_pred) <= sd
        traj_models.append(reg)
        residual_sd.append(sd)
        traj_mae[label] = float(err.mean())
        coverage[label] = float(band.mean())

    models = {
        "persistence": (persistence, x_persist),
        "static_logistic": (static, x_static),
        "wearable_hgb": (wearable, x_wear),
        "fusion_hgb": (fusion, x_fusion),
    }
    calibrators = {}
    test_proba = {}
    reports = {}
    for name, (model, x) in models.items():
        raw_val = model.predict_proba(x[masks["val"]])[:, 1]
        calibrators[name] = _fit_iso(raw_val, y[masks["val"]])
        test_proba[name] = _positive(model, calibrators[name], x[masks["test"]])
        reports[name] = classification_report(y[masks["test"]], test_proba[name], target_spec=TARGET_SPEC)

    raw_hypo = hypo.predict_proba(x_fusion[masks["val"]])[:, 1]
    hypo_cal = _fit_iso(raw_hypo, y_hypo[masks["val"]])
    hypo_test = _positive(hypo, hypo_cal, x_fusion[masks["test"]])
    hypo_report = classification_report(y_hypo[masks["test"]], hypo_test, target_spec=TARGET_SPEC)

    natural = masks["test"] & (table["walk_now"] == 0) & (table["skip_next_meal"] == 0)
    natural_reports = {}
    for name, (model, x) in models.items():
        natural_reports[name] = classification_report(
            y[natural], _positive(model, calibrators[name], x[natural]), target_spec=TARGET_SPEC
        )

    medians = {name: float(np.median(table[name][masks["train"]])) for name in FUSION_FEATURES}
    perm = _group_permutation(
        fusion,
        calibrators["fusion_hgb"],
        x_fusion[masks["test"]][:5000],
        y[masks["test"]][:5000],
        FUSION_FEATURES,
        seed,
    )
    actions = _action_shift(fusion, calibrators["fusion_hgb"], x_fusion[masks["test"]], FUSION_FEATURES)

    print("Scoring test timelines for lead time...")
    test_ids = set(splits["test"].tolist())
    ehr_test = ehr[ehr["patient_id"].isin(test_ids)]
    parts = []
    threshold = reports["fusion_hgb"]["threshold"]
    for _, row in ehr_test.iterrows():
        patient = row.to_dict()
        sim = simulate_patient(patient, days=days, enable_random_interventions=True)
        times = []
        probs = []
        n = sim["cgm"].shape[0]
        for t in range(72, n - 24, 6):
            feats = window_features(patient, sim, t)
            vec = np.array([[feats[c] for c in FUSION_FEATURES]], dtype=float)
            times.append(t)
            probs.append(float(_positive(fusion, calibrators["fusion_hgb"], vec)[0]))
        parts.append(lead_time_report(sim["cgm"], np.asarray(times), np.asarray(probs), threshold))
    lead = summarize_lead(parts, patient_days=float(len(ehr_test) * days))

    fusion_auroc = reports["fusion_hgb"]["auroc"]
    comparison = {
        name: {
            "auroc": reports[name]["auroc"],
            "delta_vs_fusion": fusion_auroc - reports[name]["auroc"],
        }
        for name in reports
    }
    metrics = {
        "task": "P(CGM glucose reaches >=180 mg/dL within 2 hours)",
        "secondary_task": "P(CGM glucose falls below 70 mg/dL within 2 hours)",
        "units": "mg/dL",
        "seed": seed,
        "n_patients": n_patients,
        "days_per_patient": days,
        "window_hours": 6,
        "horizon_hours": 2,
        "split": "patient-level 70/15/15",
        "n_windows": {name: int(mask.sum()) for name, mask in masks.items()},
        "n_patients_split": {name: int(len(ids)) for name, ids in splits.items()},
        "hyper_prevalence_test": float(y[masks["test"]].mean()),
        "hypo_prevalence_test": float(y_hypo[masks["test"]].mean()),
        "target_specificity": TARGET_SPEC,
        "models": reports,
        "naturalistic_slice_no_logged_action": natural_reports,
        "naturalistic_slice_n": int(natural.sum()),
        "hypoglycemia_fusion": hypo_report,
        "lead_time": lead,
        "trajectory_mae_mg_dl": traj_mae,
        "trajectory_residual_sd_mg_dl": {h[0]: residual_sd[i] for i, h in enumerate(horizons)},
        "trajectory_band_coverage": coverage,
        "group_permutation_auroc_drop": perm,
        "counterfactual_shift_on_rising_windows": actions,
        "fusion_minus_baseline_auroc": comparison,
        "data": "synthetic, seeded, no real patient records",
        "fusion_training_note": (
            "Fusion and hypoglycemia fits upweight a logged walk or a skipped meal by 3x "
            "so those rarer actions are not ignored. Reported test metrics are unweighted. "
            "A walk counterfactual in the dashboard resimulates steps; the meal-skip flag "
            "is the action the probability model uses directly."
        ),
    }
    bundle = {
        "version": 1,
        "seed": seed,
        "days": days,
        "static_features": STATIC_FEATURES,
        "wearable_features": WEARABLE_FEATURES,
        "context_features": CONTEXT_FEATURES,
        "fusion_features": FUSION_FEATURES,
        "driver_groups": DRIVER_GROUPS,
        "medians": medians,
        "persistence": persistence,
        "static": static,
        "wearable": wearable,
        "fusion": fusion,
        "hypo": hypo,
        "calibrators": calibrators,
        "hypo_calibrator": hypo_cal,
        "trajectory": traj_models,
        "trajectory_residual_sd": residual_sd,
        "threshold_hyper": threshold,
        "threshold_hypo": hypo_report["threshold"],
        "hyper_mgdl": HYPER_MGDL,
        "metrics": _round(metrics),
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(_round(metrics), indent=2))
        print("Saved", MODEL_PATH)
    print(json.dumps(_round({k: {"auroc": v["auroc"], "auprc": v["auprc"]} for k, v in reports.items()}), indent=2))
    print("Lead time", json.dumps(_round(lead), indent=2))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the glycemic digital twin models")
    parser.add_argument("--patients", type=int, default=360)
    parser.add_argument("--days", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    train(n_patients=args.patients, days=args.days, seed=args.seed)


if __name__ == "__main__":
    main()
