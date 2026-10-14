"""Generator tests: seed stability and physiological coupling. No real records."""

import numpy as np

from twin.ehr import generate_ehr
from twin.features import window_features, window_labels
from twin.wearable import simulate_patient


def _patient(**overrides):
    base = {
        "patient_id": "DT-TEST",
        "age": 56,
        "sex_male": 1,
        "bmi": 28.0,
        "diabetes_years": 8,
        "hba1c_pct": 8.4,
        "fasting_glucose_mg_dl": 150,
        "ldl_mg_dl": 120,
        "hdl_mg_dl": 42,
        "triglycerides_mg_dl": 170,
        "egfr_ml_min": 88,
        "sbp_mmhg": 132,
        "family_history_t2d": 1,
        "tcf7l2_alleles": 1,
        "cdkal1_alleles": 1,
        "on_metformin": 0,
        "on_sulfonylurea": 0,
        "insulin_total_daily_units": 0,
        "diet_rice_forward": 1,
        "urban": 1,
        "schedule_seed": 123456,
    }
    base.update(overrides)
    return base


def test_ehr_is_reproducible_and_uses_mg_dl():
    a = generate_ehr(12, seed=2026)
    b = generate_ehr(12, seed=2026)
    c = generate_ehr(12, seed=2027)
    assert a.equals(b)
    assert not a.equals(c)
    assert a["fasting_glucose_mg_dl"].between(90, 300).all()
    assert a["hba1c_pct"].between(6, 13).all()
    assert set(a["tcf7l2_alleles"].unique()).issubset({0, 1, 2})
    assert (a["insulin_total_daily_units"] >= 0).all()


def test_higher_hba1c_raises_mean_glucose():
    low = simulate_patient(_patient(hba1c_pct=6.8, tcf7l2_alleles=0, bmi=23, on_metformin=1), days=3)
    high = simulate_patient(_patient(hba1c_pct=11.0, tcf7l2_alleles=2, bmi=33, on_metformin=0), days=3)
    assert float(high["cgm"].mean()) > float(low["cgm"].mean()) + 30


def test_meal_and_insulin_and_walk_move_glucose():
    plain = _patient()
    treated = _patient(on_metformin=1, insulin_total_daily_units=28)
    base = simulate_patient(plain, days=2)
    meals = np.flatnonzero(
        (base["meal_carb"] >= 50)
        & (base["minute_of_day"] >= 12 * 60)
        & (base["minute_of_day"] <= 15 * 60)
    )
    assert meals.size > 0
    m = int(meals[0])
    rise = float(base["cgm"][m + 10] - base["cgm"][m])
    assert rise > 15

    ins = simulate_patient(treated, days=2)
    assert float(ins["cgm"][m : m + 28].max()) < float(base["cgm"][m : m + 28].max()) - 8

    walked = simulate_patient(plain, days=2, extra_walk_minute=(m + 4) * 5)
    assert float(walked["cgm"][m : m + 28].max()) < float(base["cgm"][m : m + 28].max()) - 8
    assert np.allclose(base["cgm"][:m], walked["cgm"][:m])


def test_features_do_not_read_future_glucose():
    patient = _patient()
    sim = simulate_patient(patient, days=2)
    t = 72 + 36
    feats = window_features(patient, sim, t)
    labels = window_labels(sim, t)
    assert abs(feats["cgm_last"] - float(sim["cgm"][t])) < 1e-4
    future = sim["cgm"].copy()
    future[t + 1 :] = 250
    sim_future = dict(sim)
    sim_future["cgm"] = future
    feats_future = window_features(patient, sim_future, t)
    for key, value in feats.items():
        assert abs(value - feats_future[key]) < 1e-6
    assert labels["y_hyper"] in (0, 1)
    assert window_labels(sim_future, t)["y_hyper"] == 1
    assert np.isfinite(list(feats.values())).all()
