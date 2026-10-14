"""Synthetic type 2 diabetes EHR generator.

Every row is drawn from a seeded random generator. Ranges are in the units
used in Indian laboratories and clinics: glucose, lipids in mg/dL, HbA1c in
percent, eGFR in mL/min/1.73 m^2. Genetic fields are simulated allele counts,
not sequenced genotypes.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from twin.physiology import derive_physiology

GIVEN_F = (
    "Aditi",
    "Meera",
    "Ananya",
    "Kavya",
    "Ishita",
    "Diya",
    "Nisha",
    "Pooja",
    "Lakshmi",
    "Asha",
    "Revathi",
    "Sneha",
    "Harini",
    "Farida",
)
GIVEN_M = (
    "Arjun",
    "Kabir",
    "Vihaan",
    "Rohan",
    "Karthik",
    "Amit",
    "Sanjay",
    "Imran",
    "Nikhil",
    "Harsh",
    "Varun",
    "Manish",
    "Pranav",
    "Yusuf",
)
SURNAMES = (
    "Iyer",
    "Sharma",
    "Reddy",
    "Kulkarni",
    "Banerjee",
    "Patel",
    "Nair",
    "Singh",
    "Joshi",
    "Das",
    "Menon",
    "Pillai",
    "Rao",
    "Mehta",
    "Choudhary",
)
CITIES = (
    ("Pune", "Maharashtra", 1),
    ("Jaipur", "Rajasthan", 1),
    ("Coimbatore", "Tamil Nadu", 1),
    ("Lucknow", "Uttar Pradesh", 1),
    ("Nagpur", "Maharashtra", 1),
    ("Indore", "Madhya Pradesh", 1),
    ("Bhubaneswar", "Odisha", 1),
    ("Mysuru", "Karnataka", 1),
    ("Nashik", "Maharashtra", 1),
    ("Madurai", "Tamil Nadu", 1),
    ("Ranchi", "Jharkhand", 0),
    ("Varanasi", "Uttar Pradesh", 0),
    ("Kolhapur", "Maharashtra", 1),
    ("Warangal", "Telangana", 0),
)


def _clip(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _one_patient(rng: np.random.Generator, index: int) -> dict:
    sex_male = int(rng.random() < 0.52)
    sex = "Male" if sex_male else "Female"
    given = GIVEN_M[int(rng.integers(0, len(GIVEN_M)))] if sex_male else GIVEN_F[int(rng.integers(0, len(GIVEN_F)))]
    surname = SURNAMES[int(rng.integers(0, len(SURNAMES)))]
    city, state, urban = CITIES[int(rng.integers(0, len(CITIES)))]

    age = int(np.clip(rng.normal(54, 10), 32, 78))
    # Adult Asian-Indian type 2 diabetes cohort: mass around BMI 27, not a European obesity curve.
    bmi = _clip(rng.normal(27.4, 3.3), 19.5, 39.0)
    height = _clip(rng.normal(169.5 if sex_male else 156.5, 6.0), 145, 190)
    weight = bmi * (height / 100.0) ** 2

    diabetes_years = _clip(rng.normal(7.5, 4.5), 0.5, 28)
    # Clinic mix: many people above 8%, a smaller group near target.
    # Mixture: most people near a treated target, a smaller group well above it.
    # That is the usual outpatient mix, and it is what makes a 180 mg/dL
    # excursion common after meals without putting every trace above the line.
    if rng.random() < 0.28:
        hba1c = _clip(rng.normal(9.8, 0.85), 8.4, 12.4)
    else:
        hba1c = _clip(rng.normal(7.5, 0.65), 6.4, 9.0)
    tcf7l2 = int(rng.binomial(2, 0.32))
    cdkal1 = int(rng.binomial(2, 0.28))
    family = int(rng.random() < (0.30 + 0.07 * tcf7l2))

    egfr = _clip(104 - 0.55 * (age - 40) - 0.7 * diabetes_years + rng.normal(0, 8), 32, 125)
    sbp = _clip(122 + 0.35 * (age - 50) + 0.7 * (bmi - 25) + rng.normal(0, 8), 98, 178)
    dbp = _clip(78 + 0.18 * (sbp - 120) + rng.normal(0, 4), 58, 110)

    # Metformin is first line. Sulfonylureas remain common in India when cost
    # constrains newer agents. Insulin is reserved here for higher HbA1c or
    # long duration, with partial overlap.
    on_metformin = int(rng.random() < 0.80)
    su_p = _clip(0.34 + 0.05 * (hba1c - 8.0), 0.18, 0.72)
    # Insulin is not reserved for the highest HbA1c. A treated person can sit
    # near target and still hypoglyce if a meal is missed.
    ins_p = _clip(0.20 + 0.04 * (hba1c - 8.0) + (0.06 if diabetes_years > 10 else 0.0), 0.16, 0.55)
    on_su = int(rng.random() < su_p)
    on_insulin = int(rng.random() < ins_p)
    if on_su and on_insulin and rng.random() < 0.65:
        on_su = 0
    insulin_units = float(np.clip(rng.normal(26, 8), 12, 54)) if on_insulin else 0.0
    if not on_metformin and not on_su and insulin_units == 0 and hba1c > 8 and rng.random() < 0.5:
        on_metformin = 1

    diet = ("rice", "wheat", "mixed")[int(rng.integers(0, 3))]
    smoking = int(rng.random() < (0.18 if sex_male else 0.04))

    draft = {
        "hba1c_pct": hba1c,
        "bmi": bmi,
        "diabetes_years": diabetes_years,
        "tcf7l2_alleles": tcf7l2,
        "egfr_ml_min": egfr,
        "on_metformin": on_metformin,
        "on_sulfonylurea": on_su,
        "insulin_total_daily_units": insulin_units,
    }
    phys = derive_physiology(draft)
    fasting = _clip(phys["setpoint_mg_dl"] * 0.90 + rng.normal(0, 8), 95, 290)
    triglycerides = _clip(138 + 16 * (hba1c - 7) + rng.normal(0, 28), 70, 420)
    hdl = _clip((46 if sex_male else 52) - 2.0 * (hba1c - 7) + rng.normal(0, 5), 24, 78)
    ldl = _clip(108 + 3.5 * (hba1c - 7) + rng.normal(0, 22), 55, 220)

    schedule_seed = int(rng.integers(0, 2**31 - 1))
    return {
        "patient_id": f"DT-IN-{index + 1:04d}",
        "display_name": f"{given} {surname}",
        "age": age,
        "sex": sex,
        "sex_male": sex_male,
        "city": city,
        "state": state,
        "urban": int(urban),
        "height_cm": round(height, 1),
        "weight_kg": round(weight, 1),
        "bmi": round(bmi, 2),
        "diabetes_years": round(diabetes_years, 1),
        "hba1c_pct": round(hba1c, 2),
        "fasting_glucose_mg_dl": round(fasting, 1),
        "ldl_mg_dl": round(ldl, 1),
        "hdl_mg_dl": round(hdl, 1),
        "triglycerides_mg_dl": round(triglycerides, 1),
        "egfr_ml_min": round(egfr, 1),
        "sbp_mmhg": round(sbp, 0),
        "dbp_mmhg": round(dbp, 0),
        "family_history_t2d": family,
        "tcf7l2_alleles": tcf7l2,
        "cdkal1_alleles": cdkal1,
        "on_metformin": on_metformin,
        "on_sulfonylurea": on_su,
        "insulin_total_daily_units": round(insulin_units, 1),
        "diet_pattern": diet,
        "diet_rice_forward": int(diet == "rice"),
        "smoking": smoking,
        "schedule_seed": schedule_seed,
        "synthetic": 1,
    }


def generate_ehr(n: int, seed: int = 2026) -> pd.DataFrame:
    """Generate ``n`` synthetic outpatient records. The same seed repeats the cohort."""
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = np.random.default_rng(seed)
    rows = [_one_patient(rng, i) for i in range(n)]
    frame = pd.DataFrame(rows)
    frame.attrs["seed"] = seed
    frame.attrs["note"] = "Synthetic cohort. Not real patients."
    return frame


def patient_dict(row: Mapping | pd.Series) -> dict:
    if isinstance(row, pd.Series):
        return row.to_dict()
    return dict(row)
