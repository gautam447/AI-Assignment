"""The committed run must show fusion ahead of the three baselines."""

import json
from pathlib import Path

import pytest

METRICS = Path(__file__).resolve().parents[1] / "artifacts" / "metrics.json"


@pytest.mark.skipif(not METRICS.exists(), reason="full training artifact is not built")
def test_fusion_beats_baselines_on_the_saved_run():
    metrics = json.loads(METRICS.read_text())
    fusion = metrics["models"]["fusion_hgb"]["auroc"]
    for name in ("persistence", "static_logistic", "wearable_hgb"):
        assert fusion > metrics["models"][name]["auroc"]
    assert metrics["hyper_prevalence_test"] > 0.05
    assert "lead_time" in metrics
    assert metrics["data"].startswith("synthetic")
