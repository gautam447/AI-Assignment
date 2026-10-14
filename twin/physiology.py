"""Map a static EHR row to the latent physiology the wearable simulator uses.

The trained model never sees these latents. It sees the chart, the sensor
window, and the meal or dose log. The latents exist so the two generators
stay coupled.
"""

from __future__ import annotations

import math
from typing import Mapping


def derive_physiology(row: Mapping) -> dict:
    """Return insulin sensitivity, glucose setpoint, dawn size, and meal scale.

    HbA1c is converted with the ADAG estimated-average-glucose equation
    (28.7 * HbA1c% - 46.7), which is in mg/dL. BMI excess is measured from
    23 kg/m^2, the WHO Asian-Indian action point, not the European 25 cutoff.
    """
    hba1c = float(row["hba1c_pct"])
    bmi = float(row["bmi"])
    years = float(row["diabetes_years"])
    tcf = float(row["tcf7l2_alleles"])
    egfr = float(row["egfr_ml_min"])
    metformin = float(row["on_metformin"])
    su = float(row["on_sulfonylurea"])
    insulin = float(row["insulin_total_daily_units"])

    eag = 28.7 * hba1c - 46.7
    bmi_excess = max(0.0, bmi - 23.0) / 10.0
    renal_penalty = max(0.0, (90.0 - egfr) / 90.0)
    si = (
        1.15
        - 0.07 * (hba1c - 7.0)
        - 0.10 * bmi_excess
        - 0.08 * tcf
        - 0.04 * (years / 10.0)
        - 0.06 * renal_penalty
        + 0.10 * metformin
        + 0.12 * su
        + 0.006 * insulin
    )
    si = float(min(1.55, max(0.38, si)))

    # Between-meal target, below the ADAG average because meals sit on top.
    # Medicines lower the target they were prescribed to lower.
    setpoint = (
        138.0
        + 14.0 * (hba1c - 7.0)
        + 3.5 * tcf
        + 5.0 * bmi_excess
        - 8.0 * metformin
        - 11.0 * su
        - 0.55 * insulin
    )
    setpoint = float(min(230.0, max(108.0, setpoint)))

    dawn = (12.0 + 7.0 * max(0.0, hba1c - 7.0) + 4.0 * tcf) * (1.0 - 0.40 * metformin)
    dawn = float(min(48.0, max(5.0, dawn)))

    # Total mg/dL added by an 80 g meal across the appearance kernel.
    # Clearance removes part of it before the peak, so the visible rise is smaller.
    meal_peak = (102.0 + 12.0 * max(0.0, hba1c - 7.0) + 6.0 * tcf) / (0.58 + 0.40 * si)
    meal_peak = float(min(150.0, max(48.0, meal_peak)))

    return {
        "eag_mg_dl": float(eag),
        "insulin_sensitivity": si,
        "setpoint_mg_dl": setpoint,
        "dawn_amp_mg_dl": dawn,
        "meal_peak_80g": meal_peak,
        "insulin_units": insulin,
        "on_sulfonylurea": su,
        "on_metformin": metformin,
    }


def clock_label(minute_of_day: int) -> str:
    minute_of_day = int(minute_of_day) % (24 * 60)
    hour, minute = divmod(minute_of_day, 60)
    return f"{hour:02d}:{minute:02d}"


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
