"""Clinician view of the synthetic glycemic twin.

Run from the repository root with `python run_demo.py`.
Every person on this screen is generated. Nothing here is a medical device.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from twin.demo import load_demo, sim_from_row
from twin.infer import load_bundle, score_window
from twin.physiology import clock_label
from twin.wearable import simulate_patient

st.set_page_config(page_title="Glycemic Digital Twin", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px;}
    h1 {font-size: 1.7rem; letter-spacing: -0.02em; margin-bottom: 0.1rem;}
    .banner {background: #0F6E6B; color: #F4F7F6; padding: 14px 18px; border-radius: 12px; margin-bottom: 14px;}
    .banner small {color: #D5E6E4;}
    .alert-hyper {background: #FEF2F2; border-left: 4px solid #C2410C; padding: 10px 12px; border-radius: 8px;}
    .alert-hypo {background: #FFFBEB; border-left: 4px solid #B45309; padding: 10px 12px; border-radius: 8px;}
    .alert-ok {background: #F0FDF4; border-left: 4px solid #3F6212; padding: 10px 12px; border-radius: 8px;}
    .fine {color: #52605E; font-size: 0.86rem;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def bundle():
    return load_bundle()


@st.cache_data
def cohort():
    ehr, series = load_demo()
    model = bundle()
    rows = []
    for i, rec in ehr.iterrows():
        patient = rec.to_dict()
        sim = sim_from_row(series, i)
        anchor = int(series["anchor"][i])
        scored = score_window(model, patient, sim, anchor)
        rows.append(
            {
                "i": i,
                "patient_id": patient["patient_id"],
                "name": patient["display_name"],
                "city": f"{patient['city']}, {patient['state']}",
                "age_sex": f"{int(patient['age'])}{'M' if int(patient['sex_male']) else 'F'}",
                "hba1c": float(patient["hba1c_pct"]),
                "p_hyper": scored["p_hyper"]["fusion_hgb"],
                "p_hypo": scored["p_hypo"],
                "alert_hyper": scored["alert_hyper"],
                "alert_hypo": scored["alert_hypo"],
            }
        )
    return ehr, series, pd.DataFrame(rows)


def meds_line(patient: dict) -> str:
    parts = []
    if int(patient["on_metformin"]):
        parts.append("metformin")
    if int(patient["on_sulfonylurea"]):
        parts.append("sulfonylurea")
    units = float(patient["insulin_total_daily_units"])
    if units > 0:
        parts.append(f"insulin {units:.0f} U/day")
    return ", ".join(parts) if parts else "no glucose-lowering medicine on the chart"


def clock_series(sim: dict) -> np.ndarray:
    start = np.datetime64("2026-03-02")
    day = np.asarray(sim["day_index"]).astype("timedelta64[D]")
    minute = np.asarray(sim["minute_of_day"]).astype("timedelta64[m]")
    return start + day + minute


def chart(patient, base_sim, alt_sim, t: int, scored: dict, show_alt: bool) -> go.Figure:
    ts = clock_series(base_sim)
    lo = max(0, t - 144)
    hi = min(len(ts) - 1, t + 24)
    fig = go.Figure()
    fig.add_hline(y=180, line_dash="dot", line_color="#C2410C", annotation_text="180", annotation_position="right")
    fig.add_hline(y=70, line_dash="dot", line_color="#B45309", annotation_text="70", annotation_position="right")
    fig.add_trace(
        go.Scatter(
            x=ts[lo : t + 1],
            y=np.asarray(base_sim["cgm"][lo : t + 1], dtype=float),
            mode="lines",
            name="CGM history",
            line=dict(color="#0F6E6B", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts[t : hi + 1],
            y=np.asarray(base_sim["cgm"][t : hi + 1], dtype=float),
            mode="lines",
            name="Simulated outcome",
            line=dict(color="#64748B", width=1.6, dash="dot"),
        )
    )
    horizons = [0, 30, 60, 90, 120]
    center = [float(base_sim["cgm"][t])] + scored["trajectory_mg_dl"]
    low = [float(base_sim["cgm"][t])] + scored["trajectory_low"]
    high = [float(base_sim["cgm"][t])] + scored["trajectory_high"]
    if show_alt and alt_sim is not None:
        shift_c, shift_lo, shift_hi = [], [], []
        for step, minutes in enumerate((6, 12, 18, 24)):
            delta = float(alt_sim["cgm"][t + minutes] - base_sim["cgm"][t + minutes])
            shift_c.append(scored["trajectory_mg_dl"][step] + delta)
            shift_lo.append(scored["trajectory_low"][step] + delta)
            shift_hi.append(scored["trajectory_high"][step] + delta)
        center = [center[0]] + shift_c
        low = [low[0]] + shift_lo
        high = [high[0]] + shift_hi
        fig.add_trace(
            go.Scatter(
                x=ts[t : hi + 1],
                y=np.asarray(alt_sim["cgm"][t : hi + 1], dtype=float),
                mode="lines",
                name="Counterfactual physiology",
                line=dict(color="#1D4ED8", width=2, dash="dash"),
            )
        )
    future_ts = [ts[t] + np.timedelta64(m, "m") for m in horizons]
    fig.add_trace(go.Scatter(x=future_ts + future_ts[::-1], y=high + low[::-1], fill="toself", fillcolor="rgba(194,65,12,0.15)", line=dict(width=0), name="±1 residual SD", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_ts, y=center, mode="lines+markers", name="Fusion trajectory", line=dict(color="#C2410C", width=2), marker=dict(size=7)))
    fig.add_vline(x=ts[t], line_dash="dash", line_color="#1C2B2A", opacity=0.45)
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.12),
        yaxis_title="Glucose (mg/dL)",
        xaxis_title="Simulated clock",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1C2B2A", size=13),
        yaxis=dict(gridcolor="#E6EEEC", range=[40, max(280, float(np.nanmax(base_sim["cgm"][lo:hi])) + 15)]),
        xaxis=dict(gridcolor="#E6EEEC"),
    )
    return fig


def main() -> None:
    try:
        model = bundle()
        ehr, series, summary = cohort()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    metrics = model["metrics"]
    st.markdown(
        f"""
<div class="banner">
  <div style="font-size:1.35rem;font-weight:650;">Glycemic digital twin</div>
  <small>Type 2 diabetes · probability CGM reaches ≥180 mg/dL in 2 hours · synthetic India cohort · seed {metrics["seed"]}</small>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Virtual patients", len(summary))
    c2.metric("Hyper alerts", int(summary["alert_hyper"].sum()))
    c3.metric("Hypo alerts", int(summary["alert_hypo"].sum()))
    c4.metric("Fusion test AUROC", f"{metrics['models']['fusion_hgb']['auroc']:.3f}")
    st.markdown(
        '<p class="fine">The list is this morning’s plan: no extra walk and no skipped meal. Alerts use the thresholds locked on the validation split at specificity 0.80. Names are generated labels, not people.</p>',
        unsafe_allow_html=True,
    )

    show = summary[["patient_id", "name", "city", "age_sex", "hba1c"]].copy()
    show["Hyper %"] = (summary["p_hyper"] * 100).round(0)
    show["Hypo %"] = (summary["p_hypo"] * 100).round(0)
    show = show.rename(
        columns={
            "patient_id": "ID",
            "name": "Name",
            "city": "City",
            "age_sex": "Age / sex",
            "hba1c": "HbA1c %",
        }
    )
    event = st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        height=280,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "HbA1c %": st.column_config.NumberColumn(format="%.1f"),
            "Hyper %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "Hypo %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
        },
        key="roster",
    )
    labels = [
        (
            f"{row.patient_id} · {row.name} · {row.city} · "
            f"hyper {row.p_hyper * 100:.0f}% · hypo {row.p_hypo * 100:.0f}%"
        )
        for row in summary.itertuples(index=False)
    ]
    table_row = None
    selection = getattr(event, "selection", None)
    rows = getattr(selection, "rows", None) if selection is not None else None
    if rows:
        table_row = int(rows[0])
    # Apply a table click only when the clicked row changes, so the dropdown
    # stays put until the next click.
    if table_row is not None and table_row != st.session_state.get("last_table_row"):
        st.session_state["last_table_row"] = table_row
        st.session_state["patient_pick"] = table_row
    if "patient_pick" not in st.session_state:
        st.session_state["patient_pick"] = 0
    selected = st.selectbox(
        "Open a virtual patient",
        options=list(range(len(labels))),
        format_func=lambda i: labels[i],
        key="patient_pick",
    )

    patient = ehr.iloc[selected].to_dict()
    base_sim = sim_from_row(series, selected)
    anchor = int(series["anchor"][selected])
    action = st.radio(
        "Counterfactual, applied at the decision minute",
        ["Current plan", "15-minute brisk walk now", "Skip the next meal"],
        horizontal=True,
    )
    alt = None
    if action == "15-minute brisk walk now":
        alt = simulate_patient(patient, days=int(base_sim["cgm"].shape[0] // 288), extra_walk_minute=anchor * 5)
        scored = score_window(model, patient, alt, anchor)
    elif action == "Skip the next meal":
        alt = simulate_patient(patient, days=int(base_sim["cgm"].shape[0] // 288), skip_meal_from_minute=anchor * 5)
        scored = score_window(model, patient, alt, anchor)
    else:
        scored = score_window(model, patient, base_sim, anchor)
    base_scored = score_window(model, patient, base_sim, anchor)

    st.markdown(
        f"### {patient['display_name']}  ·  {patient['patient_id']}  ·  {patient['city']}, {patient['state']}"
    )
    st.markdown(
        f"<p class=\"fine\">{int(patient['age'])} years · {patient['sex']} · BMI {float(patient['bmi']):.1f} · "
        f"diabetes {float(patient['diabetes_years']):.0f} years · HbA1c {float(patient['hba1c_pct']):.1f}% · "
        f"fasting glucose {float(patient['fasting_glucose_mg_dl']):.0f} mg/dL · eGFR {float(patient['egfr_ml_min']):.0f} · "
        f"TCF7L2 alleles {int(patient['tcf7l2_alleles'])} · CDKAL1 alleles {int(patient['cdkal1_alleles'])} · "
        f"family history {'yes' if int(patient['family_history_t2d']) else 'no'} · {meds_line(patient)} · "
        f"diet {patient['diet_pattern']} · decision {clock_label(int(base_sim['minute_of_day'][anchor]))}</p>",
        unsafe_allow_html=True,
    )

    hyper_pct = 100 * scored["p_hyper"]["fusion_hgb"]
    hypo_pct = 100 * scored["p_hypo"]
    base_hyper = 100 * base_scored["p_hyper"]["fusion_hgb"]
    a, b = st.columns([1.3, 1])
    with a:
        if scored["alert_hyper"]:
            st.markdown(
                f'<div class="alert-hyper"><strong>Hyperglycemia alert.</strong> Fusion probability of reaching ≥180 mg/dL in 2 hours is {hyper_pct:.0f}% '
                f"(current plan {base_hyper:.0f}%). Threshold {100 * model['threshold_hyper']:.0f}%.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alert-ok"><strong>No hyperglycemia alert.</strong> Fusion probability is {hyper_pct:.0f}% '
                f"(current plan {base_hyper:.0f}%). Threshold {100 * model['threshold_hyper']:.0f}%.</div>",
                unsafe_allow_html=True,
            )
    with b:
        if scored["alert_hypo"]:
            st.markdown(
                f'<div class="alert-hypo"><strong>Hypoglycemia alert.</strong> Probability of CGM &lt;70 mg/dL in 2 hours is {hypo_pct:.0f}%. '
                f"Threshold {100 * model['threshold_hypo']:.0f}%.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alert-ok"><strong>No hypoglycemia alert.</strong> Probability of CGM &lt;70 mg/dL in 2 hours is {hypo_pct:.0f}%.</div>',
                unsafe_allow_html=True,
            )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Persistence", f"{100 * scored['p_hyper']['persistence']:.0f}%")
    m2.metric("Static logistic", f"{100 * scored['p_hyper']['static_logistic']:.0f}%")
    m3.metric("Wearable model", f"{100 * scored['p_hyper']['wearable_hgb']:.0f}%")
    m4.metric("Fusion", f"{100 * scored['p_hyper']['fusion_hgb']:.0f}%")
    st.markdown(
        '<p class="fine">Same clock minute, four models. Persistence sees only the latest CGM point. Static logistic sees the chart and no clock. The wearable model sees 6-hour sensor aggregates. Fusion adds the meal and dose log.</p>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(chart(patient, base_sim, alt, anchor, scored, show_alt=alt is not None), use_container_width=True)
    st.markdown(
        '<p class="fine">Teal is the CGM the model may use. The dotted grey line is the simulated future under the current plan, held out of the features. The orange band is the fusion trajectory ± 1 residual standard deviation from validation. A blue dashed line is the same physiology with the selected action.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("**What is pushing the hyperglycemia risk**")
        drivers = pd.DataFrame(scored["drivers"])
        drivers["Risk contribution"] = drivers["delta_p"]
        fig = go.Figure(
            go.Bar(
                x=drivers["Risk contribution"],
                y=drivers["group"],
                orientation="h",
                marker_color=["#C2410C" if v > 0 else "#0F6E6B" for v in drivers["Risk contribution"]],
            )
        )
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Change in fusion probability vs a typical window",
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#1C2B2A", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        feats = scored["features"]
        st.markdown("**Window the fusion model just read**")
        st.write(
            {
                "Latest CGM (mg/dL)": round(feats["cgm_last"], 1),
                "1-hour slope (mg/dL per hour)": round(feats["cgm_slope_1h"], 1),
                "Carbs logged in last 2 h (g)": round(feats["carbs_last_2h"], 0),
                "Carbs planned in next 2 h (g)": round(feats["planned_carbs_next_2h"], 0),
                "Minutes since insulin": round(feats["minutes_since_insulin"], 0),
                "Recent insulin (U)": round(feats["insulin_units_recent"], 1),
                "Steps in last 30 min": round(feats["steps_30m"], 0),
                "Mean HRV RMSSD (ms)": round(feats["hrv_mean"], 1),
                "Sleep fraction of the 6 h": round(feats["sleep_frac"], 2),
            }
        )
        delta = scored["p_hyper"]["fusion_hgb"] - base_scored["p_hyper"]["fusion_hgb"]
        if action != "Current plan":
            direction = "lower" if delta < -0.005 else "higher" if delta > 0.005 else "about the same"
            st.markdown(
                f"Under **{action.lower()}**, fusion hyperglycemia risk is {direction} by {abs(delta) * 100:.1f} percentage points versus the current plan."
            )
        upcoming = float(np.asarray(base_sim["meal_carb"][anchor + 1 : anchor + 25]).sum())
        if action.startswith("Skip") and upcoming < 10:
            st.markdown(
                '<p class="fine">No carbohydrate meal is on the log in the next 2 hours, so skipping a meal has little physiology left to change.</p>',
                unsafe_allow_html=True,
            )

    with st.expander("Cohort result this model was trained on"):
        models = metrics["models"]
        table = pd.DataFrame(
            [
                {
                    "Model": name,
                    "AUROC": models[name]["auroc"],
                    "AUPRC": models[name]["auprc"],
                    "Sensitivity at specificity 0.80": models[name]["sensitivity_at_specificity"],
                    "ECE": models[name]["ece"],
                }
                for name in ("persistence", "static_logistic", "wearable_hgb", "fusion_hgb")
            ]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
        lead = metrics["lead_time"]
        st.markdown(
            f"Test hyperglycemia prevalence {metrics['hyper_prevalence_test']:.1%}. "
            f"Fusion detected {lead['detection_rate']:.0%} of usable rises to 180 mg/dL, "
            f"median lead {lead['median_lead_min']} min, "
            f"{lead['alerts_per_patient_day']} alerts per patient-day, "
            f"alert PPV {lead['alert_ppv']:.0%}."
        )
    st.markdown(
        '<p class="fine">Synthetic records only. Not for diagnosis, dosing, or clinical use. Thresholds and probabilities come from the held-out patients in artifacts/metrics.json.</p>',
        unsafe_allow_html=True,
    )


main()
