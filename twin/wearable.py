"""Wearable generator physiologically coupled to an EHR twin.

Five-minute series: CGM glucose (mg/dL), heart rate, HRV (RMSSD, ms),
steps, and sleep stage. Meals, the dawn rise, walking, short sleep, and
medicines change glucose. They are not independent noise streams.

A 15-minute brisk walk and a skipped meal can be forced at a clock minute
so the dashboard counterfactual uses the same equations as training.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from twin.physiology import derive_physiology

DT_MIN = 5
STEPS_PER_HOUR = 12
STEPS_PER_DAY = 24 * STEPS_PER_HOUR
HYPER_MGDL = 180.0
HYPO_MGDL = 70.0

# Meal appearance kernel, peak near 45 min, support 2.5 h. Sums to 1.
_MEAL_LAGS = np.arange(0, 30)
_MEAL_T = _MEAL_LAGS * DT_MIN
_MEAL_KERNEL = np.exp(-0.5 * ((_MEAL_T - 45.0) / 18.0) ** 2)
_MEAL_KERNEL = _MEAL_KERNEL / _MEAL_KERNEL.sum()


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def _add_kernel(target: np.ndarray, onset: int, amplitude: float) -> None:
    end = min(target.shape[0], onset + _MEAL_KERNEL.shape[0])
    if onset >= target.shape[0] or end <= onset:
        return
    k = _MEAL_KERNEL[: end - onset]
    target[onset:end] += amplitude * k


def _triangle(n_bins: int, start: int, peak: int, end: int, height: float) -> np.ndarray:
    """A triangular bump on the bin axis, inclusive of start, exclusive of end."""
    out = _zeros(n_bins)
    if height == 0 or end <= start:
        return out
    peak = min(max(peak, start + 1), end - 1)
    for t in range(max(0, start), min(n_bins, end)):
        if t <= peak:
            span = max(1, peak - start)
            out[t] = height * (t - start) / span
        else:
            span = max(1, end - peak)
            out[t] = height * (end - t) / span
    return out


def simulate_patient(
    patient: Mapping,
    days: int = 4,
    extra_walk_minute: int | None = None,
    skip_meal_from_minute: int | None = None,
    enable_random_interventions: bool = False,
) -> dict:
    """Simulate ``days`` of coupled wearable signals for one EHR row.

    ``schedule_seed`` on the patient fixes meals, sleep, organic activity,
    and sensor noise. Interventions are applied after those draws, so a
    counterfactual resimulation keeps the same history up to the decision.
    """
    if days < 1:
        raise ValueError("days must be at least 1")
    schedule_seed = int(patient["schedule_seed"])
    rng = np.random.default_rng(schedule_seed)
    phys = derive_physiology(patient)
    si = phys["insulin_sensitivity"]
    setpoint = phys["setpoint_mg_dl"]
    n = days * STEPS_PER_DAY
    minute = np.arange(n, dtype=np.int32) * DT_MIN
    minute_of_day = minute % (24 * 60)
    day_index = minute // (24 * 60)

    meal_carb = _zeros(n)
    insulin_units = _zeros(n)
    su_dose = _zeros(n)
    steps = _zeros(n)
    sleep = np.zeros(n, dtype=np.int8)
    bedtime = np.full(days, -1, dtype=np.int32)
    waketime = np.full(days, -1, dtype=np.int32)

    rice = float(patient.get("diet_rice_forward", 0))
    on_insulin = float(patient["insulin_total_daily_units"]) > 0
    total_insulin = float(patient["insulin_total_daily_units"])
    on_su = float(patient["on_sulfonylurea"]) > 0

    for day in range(days):
        base = day * STEPS_PER_DAY
        # Sleep window. Short nights are common enough to create next-day debt.
        bed_minute = int(np.clip(rng.normal(22 * 60 + 40, 40), 21 * 60, 24 * 60 - 5))
        sleep_hours = float(np.clip(rng.normal(6.6, 0.9), 4.2, 8.6))
        wake_abs = bed_minute + int(sleep_hours * 60)
        bedtime[day] = bed_minute
        waketime[day] = wake_abs % (24 * 60)
        _paint_sleep(sleep, base, bed_minute, int(sleep_hours * 60 / DT_MIN))

        meals = _day_meals(rng, rice)
        for name, clock_min, carbs, skip_p in meals:
            if rng.random() < skip_p:
                continue
            idx = base + int(clock_min) // DT_MIN
            if 0 <= idx < n:
                meal_carb[idx] += carbs

        if on_insulin:
            breakfast_dose = total_insulin * 0.45
            dinner_dose = total_insulin * 0.55
            for clock_min, dose in ((8 * 60, breakfast_dose), (20 * 60 + 15, dinner_dose)):
                idx = base + clock_min // DT_MIN
                if 0 <= idx < n:
                    insulin_units[idx] += dose
        if on_su:
            idx = base + (8 * 60) // DT_MIN
            if 0 <= idx < n:
                su_dose[idx] = 1.0

        # Background steps while awake, near zero in bed.
        for t in range(STEPS_PER_DAY):
            abs_t = base + t
            if abs_t >= n:
                break
            if sleep[abs_t] == 0:
                steps[abs_t] = float(rng.poisson(10))
            elif rng.random() < 0.02:
                steps[abs_t] = float(rng.poisson(4))

        # Some days include an ordinary walk. This is observed in the step
        # stream once it has started; it is not the dashboard counterfactual.
        if rng.random() < 0.50:
            if rng.random() < 0.5:
                start_min = int(np.clip(rng.normal(7 * 60, 20), 6 * 60, 8 * 60 + 30))
            else:
                start_min = int(np.clip(rng.normal(18 * 60, 25), 17 * 60, 19 * 60 + 15))
            _paint_walk(steps, base + start_min // DT_MIN, bins=int(rng.integers(4, 8)), pace=int(rng.integers(380, 520)))

    walk_now = _zeros(n)
    skip_flag = _zeros(n)
    if enable_random_interventions:
        inter = np.random.default_rng(schedule_seed ^ 0x5A17C3)
        _apply_random_interventions(
            inter, meal_carb, steps, walk_now, skip_flag, on_insulin or on_su
        )

    if extra_walk_minute is not None:
        idx = int(extra_walk_minute) // DT_MIN
        _paint_walk(steps, idx, bins=3, pace=480)
        if 0 <= idx < n:
            walk_now[idx] = 1.0
    if skip_meal_from_minute is not None:
        idx = int(skip_meal_from_minute) // DT_MIN
        _skip_meals(meal_carb, idx, horizon_bins=24)
        if 0 <= idx < n:
            skip_flag[idx] = 1.0

    meal_add = _zeros(n)
    meal_times = np.flatnonzero(meal_carb > 0)
    for onset in meal_times:
        carbs = float(meal_carb[onset])
        amp = phys["meal_peak_80g"] * (carbs / 80.0)
        nearby_insulin = float(insulin_units[max(0, onset - 2) : min(n, onset + 6)].sum())
        if nearby_insulin > 0:
            amp *= max(0.55, 0.78 - 0.008 * nearby_insulin)
        if on_su and 7 * STEPS_PER_HOUR <= (onset % STEPS_PER_DAY) <= 12 * STEPS_PER_HOUR:
            amp *= 0.82
        _add_kernel(meal_add, int(onset), amp)
        # Low sensitivity leaves more of the meal in the second hour.
        if si < 1.0 and onset + 10 < n:
            bonus = (1.0 - si) * amp * 0.08
            meal_add[onset + 10 : min(n, onset + 22)] += bonus / 12.0

    dawn = _dawn_level(minute_of_day, phys["dawn_amp_mg_dl"])
    debt = _sleep_debt(minute_of_day, day_index, bedtime, waketime, days)
    target = setpoint + dawn + debt

    med_drop = _zeros(n)
    mismatch = _zeros(n)
    unmatched = _zeros(n)
    dose_idx = np.flatnonzero(insulin_units > 0)
    for onset in dose_idx:
        units = float(insulin_units[onset])
        # Fall starts ~25 min after the dose and peaks near 90 min.
        med_drop += _triangle(n, int(onset) + 5, int(onset) + 18, int(onset) + 46, height=1.05 * min(units, 22))
        carbs_near = float(meal_carb[max(0, int(onset) - 3) : min(n, int(onset) + 16)].sum())
        if carbs_near < 25:
            lo = int(onset) + 6
            hi = min(n, int(onset) + 32)
            # max(), not +=, so insulin and a sulfonylurea do not stack into a crash.
            mismatch[lo:hi] = np.maximum(mismatch[lo:hi], min(46.0, 18.0 + 0.7 * units))
            unmatched[lo:hi] = np.maximum(unmatched[lo:hi], 2.2)
    for onset in np.flatnonzero(su_dose > 0):
        med_drop += _triangle(n, int(onset) + 8, int(onset) + 24, int(onset) + 84, height=14.0)
        carbs_near = float(meal_carb[max(0, int(onset) - 3) : min(n, int(onset) + 16)].sum())
        if carbs_near < 25:
            lo = int(onset) + 8
            hi = min(n, int(onset) + 34)
            mismatch[lo:hi] = np.maximum(mismatch[lo:hi], 36.0)
            unmatched[lo:hi] = np.maximum(unmatched[lo:hi], 2.0)
    target = target - med_drop - mismatch

    uptake = _activity_uptake(steps, si, insulin_units, su_dose)
    # Logged brisk walk: an extra, delayed disposal the meal log can name
    # before those steps show up in the past window. About 30 mg/dL.
    walk_extra = _zeros(n)
    for onset in np.flatnonzero(walk_now > 0):
        walk_extra[int(onset) : min(n, int(onset) + 8)] += 4.2
    process = rng.normal(0.0, 0.55, size=n)
    k = 0.055 * si

    true_g = _zeros(n)
    true_g[0] = float(np.clip(target[0] + rng.normal(0, 3), 80, 280))
    for t in range(1, n):
        pull = k * (true_g[t - 1] - target[t])
        true_g[t] = true_g[t - 1] + meal_add[t] - uptake[t] - walk_extra[t] - unmatched[t] - pull + process[t]
        if true_g[t] < 72 and med_drop[t] < 6 and unmatched[t] < 0.4:
            true_g[t] += 0.55 * (72 - true_g[t])
        true_g[t] = float(min(400.0, max(40.0, true_g[t])))

    ar = _zeros(n)
    ar[0] = float(rng.normal(0, 3.5))
    shock = rng.normal(0, 2.4, size=n)
    for t in range(1, n):
        ar[t] = 0.55 * ar[t - 1] + shock[t]
    cgm = np.clip(true_g + ar, 40.0, 420.0)

    hr, hrv = _heart(patient, phys, cgm, steps, sleep, meal_carb, rng)

    return {
        "cgm": cgm.astype(np.float32),
        "glucose_true": true_g.astype(np.float32),
        "hr": hr.astype(np.float32),
        "hrv_rmssd": hrv.astype(np.float32),
        "steps": steps.astype(np.float32),
        "sleep_stage": sleep.astype(np.int8),
        "minute_of_day": minute_of_day.astype(np.int16),
        "day_index": day_index.astype(np.int16),
        "meal_carb": meal_carb.astype(np.float32),
        "insulin_units": insulin_units.astype(np.float32),
        "su_dose": su_dose.astype(np.float32),
        "walk_now": walk_now.astype(np.float32),
        "skip_next_meal": skip_flag.astype(np.float32),
        "physiology": phys,
    }


def _day_meals(rng: np.random.Generator, rice: float) -> list[tuple[str, int, float, float]]:
    breakfast = int(np.clip(rng.normal(8 * 60 + 10, 25), 7 * 60, 9 * 60 + 20))
    lunch = int(np.clip(rng.normal(13 * 60 + 10, 25), 12 * 60, 14 * 60 + 30))
    snack = int(np.clip(rng.normal(17 * 60 + 10, 20), 16 * 60, 18 * 60))
    dinner = int(np.clip(rng.normal(20 * 60 + 20, 30), 19 * 60, 21 * 60 + 40))
    lunch_carbs = float(np.clip(rng.normal(95 + 18 * rice, 12), 70, 130))
    return [
        ("breakfast", breakfast, float(np.clip(rng.normal(64, 8), 45, 90)), 0.16),
        ("lunch", lunch, lunch_carbs, 0.07),
        ("snack", snack, float(np.clip(rng.normal(24, 6), 10, 40)), 0.50),
        ("dinner", dinner, float(np.clip(rng.normal(80, 10), 55, 115)), 0.14),
    ]


def _paint_sleep(sleep: np.ndarray, day_base: int, bed_minute: int, n_bins: int) -> None:
    # 90-min cycles: light, deep, light, REM. The first cycle is deeper.
    cycle = ([1] * 6 + [2] * 6 + [1] * 3 + [3] * 3)
    first = ([1] * 4 + [2] * 8 + [1] * 3 + [3] * 3)
    pattern = first + cycle * 6
    for i in range(n_bins):
        abs_min = bed_minute + i * DT_MIN
        day_offset = abs_min // (24 * 60)
        minute = abs_min % (24 * 60)
        idx = day_base + day_offset * STEPS_PER_DAY + minute // DT_MIN
        if 0 <= idx < sleep.shape[0]:
            sleep[idx] = pattern[i % len(pattern)]


def _paint_walk(steps: np.ndarray, onset: int, bins: int, pace: int) -> None:
    end = min(steps.shape[0], max(0, onset) + bins)
    for t in range(max(0, onset), end):
        steps[t] = max(steps[t], pace)


def _skip_meals(meal_carb: np.ndarray, onset: int, horizon_bins: int) -> None:
    end = min(meal_carb.shape[0], max(0, onset) + horizon_bins)
    meal_carb[max(0, onset) : end] = 0.0


def _apply_random_interventions(
    rng: np.random.Generator,
    meal_carb: np.ndarray,
    steps: np.ndarray,
    walk_now: np.ndarray,
    skip_flag: np.ndarray,
    medicated: bool,
) -> None:
    """Log a skip or a walk once per meal, on the 30-minute decision grid.

    Deciding at every grid point would cancel almost every meal. One coin
    flip per meal keeps the day intact and still teaches the action flags.
    """
    n = meal_carb.shape[0]
    meal_idx = [int(i) for i in np.flatnonzero(meal_carb >= 40)]
    for onset in meal_idx:
        if onset < 6 * STEPS_PER_HOUR or onset >= n - STEPS_PER_HOUR:
            continue
        # Flag the skip at the previous 30-min mark, before the carbs land.
        decision = onset - (onset % 6)
        if decision == onset:
            decision = max(0, onset - 6)
        if rng.random() < 0.16 and meal_carb[onset] > 0:
            skip_flag[decision] = 1.0
            meal_carb[onset] = 0.0
            continue
        if rng.random() < (0.55 if medicated else 0.40):
            # Walk starts 20-35 min after the first bite, on a decision bin,
            # so the flag is known before the steps appear in the past window.
            lag = int(rng.integers(4, 8))
            start = onset + lag
            start = start - (start % 6)
            if 6 * STEPS_PER_HOUR <= start < n - 2 * STEPS_PER_HOUR and float(steps[start : start + 3].mean()) < 80:
                _paint_walk(steps, start, bins=3, pace=480 if medicated else 450)
                walk_now[start] = 1.0


def _dawn_level(minute_of_day: np.ndarray, amp: float) -> np.ndarray:
    # Smooth bump centered at 06:15, essentially zero outside 04:00-09:00.
    md = minute_of_day.astype(np.float64)
    z = (md - (6 * 60 + 15)) / 70.0
    bump = amp * np.exp(-0.5 * z * z)
    bump[(md < 3 * 60 + 30) | (md > 9 * 60 + 30)] = 0.0
    return bump


def _sleep_debt(minute_of_day, day_index, bedtime, waketime, days: int) -> np.ndarray:
    out = _zeros(minute_of_day.shape[0])
    for day in range(days):
        if day == 0:
            hours = 6.5
        else:
            bed = int(bedtime[day - 1])
            wake = int(waketime[day - 1])
            # Wake clock is on the morning after bed. Duration crosses midnight.
            hours = ((24 * 60 - bed) + wake) / 60.0
        if hours >= 6.2:
            continue
        extra = float(np.clip(8 + (6.2 - hours) * 12, 8, 28))
        mask = (day_index == day) & (minute_of_day >= 6 * 60) & (minute_of_day <= 16 * 60)
        out[mask] += extra
    return out


def _activity_uptake(steps: np.ndarray, si: float, insulin_units: np.ndarray, su_dose: np.ndarray) -> np.ndarray:
    n = steps.shape[0]
    insulin_cover = _zeros(n)
    for onset in np.flatnonzero(insulin_units > 0):
        insulin_cover[int(onset) : min(n, int(onset) + 48)] += 1.0
    su_cover = _zeros(n)
    for onset in np.flatnonzero(su_dose > 0):
        su_cover[int(onset) : min(n, int(onset) + 96)] += 1.0
    excess = np.maximum(0.0, steps - 20.0)
    # ~450 steps in a 5-min brisk-walk bin removes on the order of 8-15 mg/dL
    # spread over the next half hour, more if insulin or a sulfonylurea is on board.
    inst = excess * 0.016 * (0.50 + si) * (1.0 + 0.75 * np.clip(insulin_cover, 0, 1) + 0.30 * np.clip(su_cover, 0, 1))
    kernel = np.array([0.18, 0.22, 0.20, 0.16, 0.12, 0.08, 0.04])
    uptake = np.convolve(inst, kernel, mode="full")[:n]
    return uptake


def _heart(patient, phys, cgm, steps, sleep, meal_carb, rng) -> tuple[np.ndarray, np.ndarray]:
    n = cgm.shape[0]
    age = float(patient["age"])
    bmi = float(patient["bmi"])
    sex_male = float(patient["sex_male"])
    tcf = float(patient["tcf7l2_alleles"])
    rest = 63 + 0.32 * (bmi - 25) + 0.10 * (age - 45) + 2.0 * sex_male
    meal_bump = np.convolve((meal_carb > 0).astype(float), np.exp(-np.arange(18) / 5.0), mode="full")[:n]
    meal_bump = 7.0 * np.clip(meal_bump, 0, 1)
    hr = rest - 7.5 * (sleep > 0) + 0.018 * steps + meal_bump
    hr = hr + rng.normal(0, 1.6, size=n)
    hr = np.clip(hr, 48, 175)

    base = 44 - 0.28 * (age - 40) - 0.40 * (bmi - 24) - 3.2 * tcf
    base = base * np.where(sleep == 2, 1.28, np.where(sleep > 0, 1.10, 1.0))
    glucose_penalty = np.clip(1.12 - (cgm - 110.0) / 380.0, 0.55, 1.25)
    hrv = base * glucose_penalty + rng.normal(0, 2.2, size=n)
    hrv = np.clip(hrv, 8, 110)
    return hr.astype(np.float64), hrv.astype(np.float64)
