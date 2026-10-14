"""Inference smoke test on a tiny seeded cohort. Does not overwrite artifacts."""

import numpy as np

from twin.ehr import generate_ehr
from twin.infer import score_window
from twin.train import train
from twin.wearable import simulate_patient


def test_tiny_model_scores_a_window():
    bundle = train(n_patients=28, days=2, seed=7, save=False)
    patient = generate_ehr(1, seed=7).iloc[0].to_dict()
    sim = simulate_patient(patient, days=2, enable_random_interventions=False)
    scored = score_window(bundle, patient, sim, t=180)
    for name, prob in scored["p_hyper"].items():
        assert 0.0 <= prob <= 1.0, name
    assert 0.0 <= scored["p_hypo"] <= 1.0
    assert len(scored["trajectory_mg_dl"]) == 4
    assert len(scored["drivers"]) >= 5
    assert np.isfinite(scored["trajectory_mg_dl"]).all()
    # Fusion is allowed to see the static chart and the sensor window.
    assert "hba1c_pct" in bundle["fusion_features"]
    assert "cgm_last" in bundle["fusion_features"]
    assert "planned_carbs_next_2h" in bundle["fusion_features"]
    assert "planned_carbs_next_2h" not in bundle["wearable_features"]
    assert "hba1c_pct" not in bundle["wearable_features"]
