#!/usr/bin/env python3
"""Generate the demo cohort, train or load the fusion model, and open the dashboard.

One command from a fresh clone:

    python run_demo.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ensure_dependencies() -> None:
    try:
        import joblib  # noqa: F401
        import pandas  # noqa: F401
        import plotly  # noqa: F401
        import sklearn  # noqa: F401
        import streamlit  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the glycemic digital twin demo")
    parser.add_argument("--retrain", action="store_true", help="Rebuild the model even if an artifact exists")
    parser.add_argument("--train-only", action="store_true", help="Train and write the demo cohort, do not serve")
    parser.add_argument("--check", action="store_true", help="Score one demo window and print the fusion probability")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--patients", type=int, default=320, help="Training cohort size when a model is built")
    parser.add_argument("--days", type=int, default=3, help="Simulated days per training patient")
    args = parser.parse_args()

    ensure_dependencies()
    sys.path.insert(0, str(ROOT))
    from twin.demo import DEMO_SERIES, build_demo
    from twin.infer import load_bundle, score_window
    from twin.train import MODEL_PATH, train

    if args.retrain or not MODEL_PATH.exists():
        train(n_patients=args.patients, days=args.days, seed=2026)
    bundle = load_bundle()
    if args.retrain or not DEMO_SERIES.exists():
        print("Building the 12-patient demo cohort...")
        build_demo(bundle, n=12, days=3, seed=11, pool=36)

    if args.check or args.train_only:
        from twin.demo import load_demo, sim_from_row

        ehr, series = load_demo()
        patient = ehr.iloc[0].to_dict()
        sim = sim_from_row(series, 0)
        scored = score_window(bundle, patient, sim, int(series["anchor"][0]))
        metrics = bundle["metrics"]["models"]["fusion_hgb"]
        print(
            f"{patient['patient_id']} fusion P(>=180 in 2h)={scored['p_hyper']['fusion_hgb']:.3f} "
            f"P(<70)={scored['p_hypo']:.3f}"
        )
        print(
            f"Held-out fusion AUROC={metrics['auroc']:.3f} AUPRC={metrics['auprc']:.3f} "
            f"sens@spec0.80={metrics['sensitivity_at_specificity']:.3f} ECE={metrics['ece']:.3f}"
        )
        if args.train_only or args.check:
            if args.check and not args.train_only:
                return
            if args.train_only:
                return

    dashboard = ROOT / "twin" / "dashboard.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
