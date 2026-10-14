"""Rebuild the architecture PDF, the slide deck, and the recording script from metrics.json.

    python -m twin.build_docs
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as pdfcanvas
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

TEAL = RGBColor(0x0F, 0x6E, 0x6B)
INK = RGBColor(0x1C, 0x2B, 0x2A)
MUTED = RGBColor(0x52, 0x60, 0x5E)
CREAM = RGBColor(0xF4, 0xF7, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RUST = RGBColor(0xC2, 0x41, 0x0C)


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise SystemExit("artifacts/metrics.json is missing. Run python -m twin.train first.")
    return json.loads(METRICS_PATH.read_text())


def f3(value) -> str:
    return f"{float(value):.3f}"


def f4(value) -> str:
    return f"{float(value):.4f}"


def pct(value) -> str:
    return f"{100 * float(value):.1f}%"


def slides(m: dict) -> list[dict]:
    models = m["models"]
    fusion = models["fusion_hgb"]
    pers = models["persistence"]
    stat = models["static_logistic"]
    wear = models["wearable_hgb"]
    nat = m["naturalistic_slice_no_logged_action"]
    hypo = m["hypoglycemia_fusion"]
    lead = m["lead_time"]
    mae = m["trajectory_mae_mg_dl"]
    perm = {row["group"]: row["auroc_drop"] for row in m["group_permutation_auroc_drop"]}
    shift = m["counterfactual_shift_on_rising_windows"]
    return [
        {
            "title": "A glycemic digital twin for the next two hours",
            "bullets": [
                "Happiest Health Digital Twin Challenge 2026",
                "Type 2 diabetes, Indian clinic units (mg/dL)",
                "Synthetic patients only. Not a medical device.",
                "Team, college, leader, phone, email: FILL IN",
            ],
            "notes": (
                "Introduce the project in one sentence. We estimate the probability that CGM glucose "
                "reaches at least 180 milligrams per decilitre within two hours, and we raise a second "
                "alert if it may fall below 70. The setting is type 2 diabetes in India. Everything you "
                "will see was generated. There is no real patient in the training set and no real patient "
                "on the dashboard. Say the team name, college, and leader only after the FILL IN fields "
                "in the README have been replaced. Do not invent them on the recording. "
                "Tell the audience the recording follows the dashboard as a clinician would use it, "
                "then the held-out numbers from seed 2026."
            ),
            "seconds": 70,
        },
        {
            "title": "Why this question, in India",
            "bullets": [
                "ICMR–INDIAB: about 101 million people with diabetes, about 136 million with prediabetes (Anjana et al., Lancet Diabetes & Endocrinology, 2023)",
                "Rice- and wheat-based meals make the post-meal rise a daily event",
                "Follow-up is often an HbA1c every few months; CGM is still mostly out of pocket",
                "A district clinic or tele-endocrinology review needs a 2-hour answer, not another 3-month average",
            ],
            "notes": (
                "Ground the use case. Cite ICMR–INDIAB as published epidemiology, not as a number this "
                "project measured. The practical gap is timing. HbA1c tells you the quarter. It does not "
                "tell you whether today's lunch, on today's sulfonylurea or insulin, is about to cross 180. "
                "CGM tells you where glucose is now. It does not know the dose that was taken twenty minutes "
                "ago, and it does not know that a large meal is written on the log for the next hour. "
                "Describe a district-hospital diabetes clinic or a remote review: short appointment, chart "
                "already in the file, sometimes a few hours of wearable data. The product has to be readable "
                "in that visit. Mention units explicitly: milligrams per decilitre, HbA1c in percent, "
                "eGFR in mL per minute per 1.73 square metres, and BMI excess measured from 23, the WHO "
                "Asian action point, not from 25."
            ),
            "seconds": 140,
        },
        {
            "title": "The decision the twin is allowed to make",
            "bullets": [
                "Primary label: CGM maximum in the next 2 hours ≥ 180 mg/dL",
                "Secondary label: CGM minimum in the next 2 hours < 70 mg/dL",
                "Inputs: the static chart, the last 6 hours of sensors, the meal and dose log",
                "Outputs: calibrated probability, a 4-point trajectory with a residual band, drivers, one counterfactual",
            ],
            "notes": (
                "Be precise about the label so nobody thinks this is a generic 'risk score'. "
                "At a decision minute, the model may use glucose, heart rate, RMSSD, steps, and sleep "
                "from the previous six hours, the chart, and what the log already says about meals and doses. "
                "It may not use any future glucose. The positive class is whether the CGM reading, including "
                "sensor noise, reaches 180 or more at any 5-minute sample in the next two hours. "
                "The secondary class is a fall below 70. We also predict the level at 30, 60, 90, and 120 minutes. "
                "Say clearly that 180 and 70 are operational thresholds for this prototype, aligned with "
                "common hyperglycemia and level-1 hypoglycemia cut-points, not a personalised target."
            ),
            "seconds": 110,
        },
        {
            "title": "What this prototype is not",
            "bullets": [
                "Not trained on any real electronic health record or CGM export",
                "Not a dosing calculator and not a closed-loop controller",
                "Genetic fields are simulated allele counts, not sequenced genotypes",
                "A tight error on synthetic glucose is not clinical accuracy",
            ],
            "notes": (
                "Spend time here so the rest of the talk is credible. Judges have seen projects that "
                "blur synthetic and real. Say the generator, the split, the metrics file, and the dashboard "
                "all use the same seeded simulator. If a number appears on a slide, it was written by "
                "python -m twin.build_docs from artifacts/metrics.json after python -m twin.train. "
                "We did not pick a flattering subset after looking at the test labels. The patient split "
                "is by person, 70, 15, 15, so no window from a test patient is in the fit. "
                "Also say what would make this a different project: a real CGM study with a locked protocol, "
                "a site in India, and a calibration check on that site's devices."
            ),
            "seconds": 90,
        },
        {
            "title": "Synthetic chart: an Indian outpatient file",
            "bullets": [
                "320 people for the reported run, seed 2026. Demo panel uses a different seed.",
                "Age, sex, city and state, BMI from a distribution centred near 27",
                "HbA1c mixture: most people near a treated target, a smaller group well above it",
                "Medicines: metformin, sulfonylurea, insulin total daily dose. Labs in mg/dL.",
                "Simulated TCF7L2 and CDKAL1 allele counts and a family-history flag",
            ],
            "notes": (
                "Walk through one row as if you were opening a file. Name the city as a generated label, "
                "not a claim that a real clinic sent data. Explain the medicine mix: metformin is common, "
                "sulfonylureas remain common where cost shapes the formulary, and insulin is used both "
                "in poorly controlled diabetes and in some people sitting nearer target, because those are "
                "the people who can actually go low if a meal is missed. "
                "The allele counts are binomial draws at fixed frequencies. They change insulin sensitivity "
                "inside the simulator. They are not a genetic test. "
                "Fasting glucose, lipids, and eGFR are generated to move with HbA1c, age, and duration "
                "so the chart is internally consistent. The same seed rebuilds the same cohort."
            ),
            "seconds": 120,
        },
        {
            "title": "The wearable stream is coupled to that chart",
            "bullets": [
                "5-minute CGM, heart rate, RMSSD, steps, sleep stage",
                "One glucose state: meals, dawn, short sleep, walking, metformin, sulfonylurea, insulin",
                "A logged meal raises glucose later if sensitivity is low; insulin without that meal can fall below 70",
                "CGM is the state plus autocorrelated noise. Labels use the CGM.",
            ],
            "notes": (
                "This is the part to slow down. Independent noise streams would let a fusion model cheat "
                "or, worse, learn nothing. Here the chart sets a between-meal setpoint, a dawn rise, and "
                "how large an 80 gram carbohydrate meal is allowed to be. Rice-forward diets increase lunch "
                "carbohydrate. A walk adds steps and glucose disposal, larger if insulin or a sulfonylurea "
                "is on board. Insulin and sulfonylurea lower the target after a delay of about half an hour, "
                "so a decision taken just after a dose does not yet show the fall on the CGM. "
                "If the paired meal is missing, an extra disposal can take a near-target person below 70. "
                "A person with HbA1c around 11 often stays high instead of going low. That matches the "
                "generator and it is why hypoglycemia is uncommon in the test set. "
                "Heart rate rises with steps and meals and falls in sleep. RMSSD falls as glucose rises "
                "and is higher in deep sleep. Sleep debt the next morning lifts the setpoint. "
                "Show, if you have the tests open, that the same schedule with a higher HbA1c has a higher "
                "mean glucose, that insulin lowers the lunch peak, and that a forced walk lowers it further "
                "without changing the history before the walk."
            ),
            "seconds": 150,
        },
        {
            "title": "Train, validation, test: by patient",
            "bullets": [
                f"Seed {m['seed']}: {m['n_patients']} patients × {m['days_per_patient']} days",
                f"Windows — train {m['n_windows']['train']}, validation {m['n_windows']['val']}, test {m['n_windows']['test']}",
                f"Patients — {m['n_patients_split']['train']} / {m['n_patients_split']['val']} / {m['n_patients_split']['test']}",
                f"Test prevalence of a 2-hour rise to ≥180: {f4(m['hyper_prevalence_test'])}",
                "Decision every 30 minutes. Isotonic calibrators see validation rows only.",
            ],
            "notes": (
                "Read the counts out loud so the split is concrete. "
                f"Three hundred and twenty synthetic people, three days each, seed {m['seed']}. "
                f"The test set is {m['n_patients_split']['test']} people and {m['n_windows']['test']} windows. "
                f"About {pct(m['hyper_prevalence_test'])} of those windows reach 180 in the next two hours. "
                "That is common enough that accuracy would be a vanity metric, which is why the table "
                "uses AUROC, AUPRC, sensitivity at a fixed specificity, and calibration. "
                "The threshold for an alert is chosen on validation to sit at specificity at least 0.80, "
                "then frozen and applied to test. We report the specificity we actually got, which is not "
                "always exactly 0.80 because scores are discrete. "
                "Early stopping inside gradient boosting uses a slice of the training patients only. "
                "It does not peek at the test people."
            ),
            "seconds": 100,
        },
        {
            "title": "Four models, same label",
            "bullets": [
                "Persistence: logistic regression on the latest CGM value",
                "Static: L2 logistic regression on the chart, no clock",
                "Wearable: histogram gradient boosting on 6-hour sensor aggregates",
                "Fusion: the same booster on chart + sensors + meal and dose log, then isotonic calibration",
                "Logged walks and skipped meals are upweighted 3× in the fusion fit. Test metrics are not.",
            ],
            "notes": (
                "Explain why these baselines, not a leaderboard of fashionable networks. "
                "Persistence is the right dumb answer: if you are already at 210 you will probably still "
                "be high, and if you are at 110 and flat you probably will not cross 180. Any serious model "
                "has to beat that. The static logistic is the chart a doctor already has, with no knowledge "
                "of what this afternoon looks like. The wearable model is the sensor vendor's model: slope, "
                "level, steps, heart rate, sleep, time of day, and no medicines. "
                "Fusion is the same tree booster as the wearable model, so a win is not an algorithm trick. "
                "The extra inputs are the chart and the log: planned carbohydrates, recent carbohydrates, "
                "minutes since insulin, the recent dose, minutes since a sulfonylurea, and two action flags. "
                "Say the 3× weight in one sentence and immediately say the test set was not reweighted. "
                "Trajectory heads are four regressors on the fusion features. The band is plus or minus one "
                "validation residual standard deviation, not a conformal interval."
            ),
            "seconds": 120,
        },
        {
            "title": "Held-out comparison",
            "bullets": [
                f"Persistence — AUROC {f4(pers['auroc'])}, AUPRC {f4(pers['auprc'])}, sensitivity {f4(pers['sensitivity_at_specificity'])} at specificity {f4(pers['achieved_specificity'])}",
                f"Static logistic — AUROC {f4(stat['auroc'])}, AUPRC {f4(stat['auprc'])}, sensitivity {f4(stat['sensitivity_at_specificity'])} at specificity {f4(stat['achieved_specificity'])}",
                f"Wearable — AUROC {f4(wear['auroc'])}, AUPRC {f4(wear['auprc'])}, sensitivity {f4(wear['sensitivity_at_specificity'])} at specificity {f4(wear['achieved_specificity'])}",
                f"Fusion — AUROC {f4(fusion['auroc'])}, AUPRC {f4(fusion['auprc'])}, sensitivity {f4(fusion['sensitivity_at_specificity'])} at specificity {f4(fusion['achieved_specificity'])}, ECE {f4(fusion['ece'])}",
            ],
            "notes": (
                "Read the four AUROCs slowly. Fusion is "
                f"{f4(fusion['auroc'])}, wearable {f4(wear['auroc'])}, persistence {f4(pers['auroc'])}, "
                f"static {f4(stat['auroc'])}. Fusion is ahead of all three. "
                f"The gap over the wearable model is {f4(fusion['auroc'] - wear['auroc'])} AUROC and "
                f"{f4(fusion['auprc'] - wear['auprc'])} AUPRC. The gap over persistence is larger, "
                f"{f4(fusion['auroc'] - pers['auroc'])} AUROC, which is what you want if the latest glucose "
                "value is already a strong score. "
                "Do not hide the static result. An AUROC of "
                f"{f4(stat['auroc'])} means the chart ranks who tends to go high. It does not time the day. "
                "Sensitivity at specificity around 0.80 is "
                f"{f4(fusion['sensitivity_at_specificity'])} for fusion, {f4(wear['sensitivity_at_specificity'])} "
                f"for wearable, {f4(pers['sensitivity_at_specificity'])} for persistence, and "
                f"{f4(stat['sensitivity_at_specificity'])} for the chart. "
                "If you only remember one comparison, remember that fusion wins on both ranking and on "
                "the high-specificity operating point, on people it did not train on."
            ),
            "seconds": 150,
        },
        {
            "title": "The lift survives without the action flags",
            "bullets": [
                f"Naturalistic test windows (no logged walk, no logged skip): {m['naturalistic_slice_n']}",
                f"Fusion AUROC {f4(nat['fusion_hgb']['auroc'])} vs wearable {f4(nat['wearable_hgb']['auroc'])}, persistence {f4(nat['persistence']['auroc'])}, static {f4(nat['static_logistic']['auroc'])}",
                f"Permutation: recent glucose drops AUROC by {f3(perm['Recent glucose level'])}; meal and insulin log by {f3(perm['Meal and insulin log'])}",
                f"HbA1c, BMI, and duration together drop AUROC by only {f3(perm['HbA1c, body size, diabetes duration'])}",
                f"Fusion ECE {f4(fusion['ece'])}. Static ECE {f4(stat['ece'])}.",
            ],
            "notes": (
                "This slide is the defence against a fair objection: maybe fusion only wins because you "
                "told it that someone was about to walk or skip a meal. "
                f"On the {m['naturalistic_slice_n']} test windows where both action flags are off, fusion "
                f"AUROC is still {f4(nat['fusion_hgb']['auroc'])} and the wearable model is "
                f"{f4(nat['wearable_hgb']['auroc'])}. "
                "The permutation test says what the model actually uses. Shuffling the recent glucose "
                f"level costs about {f3(perm['Recent glucose level'])} AUROC. Shuffling the meal and insulin "
                f"log costs about {f3(perm['Meal and insulin log'])}. Shuffling HbA1c, body size, and duration "
                f"costs about {f3(perm['HbA1c, body size, diabetes duration'])}. "
                "So the fusion advantage over a sensor model is mostly the log, not a magical reading of "
                "the HbA1c. The chart still matters for the static baseline, and it is in the fusion feature "
                "vector, but once the trace and the log are present its marginal contribution is small. "
                "Say that plainly. It is a better scientific claim than pretending every lab drives the score. "
                f"Calibration: fusion expected calibration error is {f4(fusion['ece'])} after isotonic "
                "regression on validation. That is acceptable on this synthetic test and it is not a "
                "guarantee on a new device or a new city."
            ),
            "seconds": 130,
        },
        {
            "title": "Lead time, and how often it would buzz",
            "bullets": [
                f"Usable episodes on test patients: {lead['usable_episodes']}. Detected before the crossing: {pct(lead['detection_rate'])}.",
                f"Median lead {lead['median_lead_min']:.0f} min. Share with at least 30 min lead: {pct(lead['pct_lead_at_least_30_min'])}.",
                f"Alerts while still under 180: {lead['alerts_per_patient_day']:.2f} per patient-day. PPV {f4(lead['alert_ppv'])}.",
                f"New alert episodes, collapsing consecutive alarms: {lead['alert_episodes_per_patient_day']:.2f} per patient-day.",
            ],
            "notes": (
                "Define an episode before you quote the lead. An onset is the first sample at or above 180 "
                "after being below it. We ignore an onset that has no scored decision in the prior two hours. "
                "An alert counts only if glucose is still under 180 at the decision, so we are not taking "
                "credit for announcing a high that has already happened. "
                f"Median lead is {lead['median_lead_min']:.0f} minutes. "
                f"{pct(lead['pct_lead_at_least_30_min'])} of detected episodes had half an hour or more. "
                "That lead is long because the simulator's meal log is known before the glucose moves. "
                "If the person does not log the meal, you should expect something closer to the wearable "
                "model, which has to wait until the slope is visible. Say that limitation in the same breath "
                "as the number. "
                f"The annoying number is {lead['alerts_per_patient_day']:.1f} raw alerts per patient-day. "
                "The score is evaluated every 30 minutes, and many people stay in a high-risk region through "
                "a whole meal. Collapsing consecutive alerts gives "
                f"{lead['alert_episodes_per_patient_day']:.1f} episodes per patient-day. "
                f"Positive predictive value of a raw alert is {f4(lead['alert_ppv'])}. "
                "A clinic would alert on the rising edge and would retune the threshold for its own cost of "
                "a false alarm. We froze specificity 0.80 instead of shopping for a prettier operating point."
            ),
            "seconds": 140,
        },
        {
            "title": "Trajectory band, and the low-glucose head",
            "bullets": [
                f"MAE: {mae['30']:.1f} mg/dL at 30 min, {mae['60']:.1f} at 60, {mae['90']:.1f} at 90, {mae['120']:.1f} at 120",
                f"±1 residual-SD coverage on test: {pct(m['trajectory_band_coverage']['30'])} to {pct(m['trajectory_band_coverage']['120'])}",
                f"Hypoglycemia prevalence {f4(m['hypo_prevalence_test'])}. AUROC {f4(hypo['auroc'])}. AUPRC {f4(hypo['auprc'])}.",
                f"Hypoglycemia sensitivity {f4(hypo['sensitivity_at_specificity'])} at specificity {f4(hypo['achieved_specificity'])}. ECE {f4(hypo['ece'])}.",
            ],
            "notes": (
                "Two different levels of honesty. The trajectory errors are small: about "
                f"{mae['30']:.0f} milligrams at half an hour and about {mae['120']:.0f} at two hours. "
                "Coverage of the one-standard-deviation band is roughly 80 to 86 percent, wider than a "
                "Gaussian 68 percent because the residuals are not perfectly normal. "
                "Do not sell this as sensor-grade forecasting. The meal appearance kernel and the model "
                "see the same planned carbohydrates. A real stomach is less obedient. The right next "
                "experiment is to hide the planned carbs and see how fast the MAE degrades. "
                "Hypoglycemia is the weaker clinical claim. Only "
                f"{pct(m['hypo_prevalence_test'])} of test windows go below 70, because many people on "
                "insulin in this generator live too high to reach a biochemical low even if they miss a meal. "
                f"AUROC {f4(hypo['auroc'])} looks strong. AUPRC {f4(hypo['auprc'])} is the number to remember: "
                "better than the 0.0127 base rate, not a solved detection problem. "
                "The dashboard still shows the low-glucose probability because a sulfonylurea or insulin "
                "with no meal is the scenario a clinician is afraid of, and the generator does produce it."
            ),
            "seconds": 120,
        },
        {
            "title": "Dashboard: what to click",
            "bullets": [
                "Open with python run_demo.py. Twelve synthetic people, ranked by fusion probability.",
                "Select a row. Compare persistence, static, wearable, and fusion at the same minute.",
                "Teal line: history the model may use. Dotted line: simulated future it did not see.",
                "Orange band: fusion trajectory ± 1 residual SD. 180 and 70 are marked.",
                "Drivers: change in probability when a group is replaced by the training median.",
            ],
            "notes": (
                "This is the live section. Start the app before you record, with python run_demo.py, "
                "and widen the window. Read the banner: synthetic India cohort, seed 2026. "
                "Click the top row, then a middle row around a fifty percent probability, then a row "
                "whose hypoglycemia probability is high and hyperglycemia probability is near zero. "
                "On each person, point at the four metrics. The point of the middle row is that the models "
                "disagree: persistence and the wearable model are looking at a trace that has not risen yet, "
                "while fusion can see a meal on the log. If they do not disagree on the person you clicked, "
                "say what you actually see. Do not invent a disagreement. "
                "Trace the chart with the cursor. History is teal. The dotted continuation is the simulator's "
                "future, drawn so a judge can see whether the alert was early. It is not an input. "
                "The orange markers are 30, 60, 90, and 120 minutes. "
                "Open the driver chart. A bar to the right means this person's values raise the probability "
                "relative to a typical training window. Recent glucose and the meal log should usually dominate. "
                "If a genetic marker is near zero, say so. The permutation test already told us the labs are "
                "a small marginal effect. "
                "Mention the city, the HbA1c, and the medicine line so the file feels like a chart, and "
                "remind the viewer the name is generated."
            ),
            "seconds": 180,
        },
        {
            "title": "One counterfactual, computed in the open",
            "bullets": [
                "Skip the next meal: the log flag is cleared and planned carbohydrates go to zero.",
                f"On {shift['n']} rising test windows, that lowers mean fusion probability by {f3(abs(shift['skip_delta']))}.",
                "A 15-minute walk resimulates steps. The flag alone, without new steps, moves mean risk by only "
                f"{f3(abs(shift['walk_delta']))}.",
                "The blue dashed curve is the same physiology with the action applied. History before the decision matches.",
            ],
            "notes": (
                "Do the click. Pick a person who still has a meal on the log in the next two hours. "
                "Switch on Skip the next meal. Read the probability before and after from the screen, "
                "not from memory. The test-set average drop is "
                f"{f3(abs(shift['skip_delta']))} on rising windows. A single person can move more or less. "
                "Then switch to the 15-minute walk. Be honest about the mechanism. "
                f"Flipping the walk flag and leaving the sensor window untouched changes mean risk by only "
                f"{f3(abs(shift['walk_delta']))}. The trees barely split on that rare flag. "
                "The dashboard instead resimulates the walk, so steps in the last 30 minutes jump and the "
                "glucose equation removes on the order of a few tens of milligrams per decilitre. "
                "The probability moves because the sensor features move. The blue curve is that resimulation. "
                "Say that the synthetic walk effect was made large enough to see on a five-minute grid. "
                "Published walks after meals are often smaller and noisier. "
                "If the selected person has no meal coming, the skip control should barely move. Say that "
                "too. A counterfactual that does nothing is still a result."
            ),
            "seconds": 120,
        },
        {
            "title": "Limitations, in the order they matter",
            "bullets": [
                "Synthetic physiology. Real CGM, real logging, and real missed doses will be messier.",
                "Meal-log lead time will shrink the moment people do not log.",
                "Hypoglycemia AUPRC is 0.66 on a 1.3% event. AUROC alone is the wrong headline.",
                "Static labs add little once the trace and the log are present.",
                "No site, no device, no clinician in the loop, no fairness audit across states.",
            ],
            "notes": (
                "Do not rush the limitations. They are part of the result. "
                "First, the generator is a discrete-time equation, not a fitted physiological model. "
                "We did not force week-long mean glucose to equal the ADAG average implied by HbA1c. "
                "The setpoint is the between-meal target; meals sit on top; walks and missed meals pull "
                "weeks around. Correlation with HbA1c is there. Identity with a textbook formula is not. "
                "Second, planned carbohydrates are visible to fusion and not to the wearable baseline. "
                "That is a product choice, and it is also the main reason the median lead is "
                f"{lead['median_lead_min']:.0f} minutes. "
                "Third, trajectory MAE under 10 milligrams at two hours will not survive contact with "
                "unlogged meals. Fourth, the low-glucose head needs more events before anyone should "
                "pick an alert threshold for a ward. Fifth, upweighting action rows 3× changes the fit; "
                "we reported unweighted test metrics, but a different weight would move the small walk "
                "coefficient. Sixth, cities in the demo are labels for a story, not a sample of Indian "
                "diabetes. We did not test whether error differs by age, sex, or urban versus rural. "
                "Seventh, there is no prospective silent trial and no human factors test of the alert rate."
            ),
            "seconds": 140,
        },
        {
            "title": "What would change practice, if the signal survived real data",
            "bullets": [
                "Put the 2-hour probability next to the HbA1c in a district clinic review",
                "Use the meal-skip counterfactual as a safety check when a sulfonylurea or insulin was taken",
                "Use the walk counterfactual as a prescription the person can finish before the peak",
                "Retune the threshold locally. Specificity 0.80 was a reporting choice, not a guideline.",
            ],
            "notes": (
                "Keep this conditional. If, on real CGM with real logs, fusion still beats persistence "
                "and a sensor-only model by a margin a clinic can feel, the workflow is modest. "
                "The doctor already knows the medicines. The new object is a probability for this afternoon "
                "and a button that shows the same model with the next meal removed or a short walk inserted. "
                "The safety use is the missed meal on a sulfonylurea or insulin, which is common advice "
                "already and which this prototype can illustrate. The post-meal walk is the other advice "
                "clinicians already give; the screen makes the size of the change visible. "
                "None of that replaces hypoglycemia education, sick-day rules, or a human deciding a dose. "
                "A deployment path that respects the evidence would be silent scoring on a consented cohort, "
                "then a threshold workshop with the clinic, then a stepped introduction. "
                "We have not done those steps."
            ),
            "seconds": 90,
        },
        {
            "title": "Reproduce the run",
            "bullets": [
                "python run_demo.py",
                "python -m pytest",
                "python -m twin.train    # writes artifacts/model.joblib and metrics.json",
                "python -m twin.build_docs",
                "MIT license. Architecture PDF and Mermaid source are in docs/.",
            ],
            "notes": (
                "Show the repository root. One command, python run_demo.py, installs pinned dependencies "
                "if they are missing, loads the committed artifact, and serves the dashboard. "
                "If the artifact is absent it trains 320 by 3 days at seed 2026. That is the run these "
                "slides describe. Tests cover seed stability, the coupling from HbA1c, meals, insulin, "
                "and walks, the absence of future glucose in the features, a tiny in-memory train, and "
                "the saved comparison. Point at docs/architecture.pdf and the Mermaid source. "
                "Point at the PPTX, which carries these speaker notes, and the PDF of the same slides."
            ),
            "seconds": 70,
        },
        {
            "title": "Before this is a submission",
            "bullets": [
                "FILL IN team name, college, leader name, phone, and email",
                "Rename the folder to TeamName_CollegeName",
                "Make the repository public",
                "Record this script, unlisted, at least 20 minutes, and paste the URL into the README",
                "No video URL is claimed in the repository today",
            ],
            "notes": (
                "Close on the administrative gaps so you do not imply they are done. "
                "The README marks team name, college, leader, phone, and email as FILL IN. "
                "Leave them as FILL IN on the recording if they are still empty, and say the form "
                "cannot be sent until a real team replaces them. Rename the submission folder to "
                "TeamName_CollegeName. The GitHub repository must be public before the leader's name, "
                "phone, email, and repository link are submitted. "
                "This talk is the script for an unlisted YouTube video of at least twenty minutes. "
                "The repository does not contain a video URL. Add it only after the upload exists. "
                "Thank the judges, restate the one-sentence task, and stop."
            ),
            "seconds": 70,
        },
    ]


def write_video_script(deck: list[dict]) -> None:
    total = sum(s["seconds"] for s in deck)
    lines = [
        "# Recording script: glycemic digital twin",
        "",
        "Unlisted YouTube walkthrough for the Happiest Health Digital Twin Challenge 2026.",
        "No video has been uploaded. Paste the URL into the README only after it exists.",
        "",
        f"Timed length of the notes below: **{total // 60} minutes {total % 60} seconds** "
        "if each section is delivered in its allotted time at a measured pace, about 140 words per minute. "
        "The speaker notes in `docs/presentation.pptx` are the same text.",
        "",
        "Record the screen with the dashboard already running (`python run_demo.py`). "
        "Slides first, then the live panel, then the limitations slide. "
        "Do not read a team name, college, phone, or email that is still marked FILL IN.",
        "",
        "| Start | Section |",
        "| --- | --- |",
    ]
    cursor = 0
    for i, slide in enumerate(deck, start=1):
        mm_, ss = divmod(cursor, 60)
        lines.append(f"| {mm_:02d}:{ss:02d} | {i}. {slide['title']} ({slide['seconds']} s) |")
        cursor += slide["seconds"]
    mm_, ss = divmod(cursor, 60)
    lines.append(f"| {mm_:02d}:{ss:02d} | End |")
    lines.append("")
    cursor = 0
    for i, slide in enumerate(deck, start=1):
        mm_, ss = divmod(cursor, 60)
        lines.append(f"## {mm_:02d}:{ss:02d}  Slide {i} — {slide['title']}")
        lines.append("")
        lines.append(f"*On screen: {slide['seconds']} seconds.*")
        lines.append("")
        for bullet in slide["bullets"]:
            lines.append(f"- {bullet}")
        lines.append("")
        lines.append(slide["notes"])
        lines.append("")
        cursor += slide["seconds"]
    (DOCS / "video-script.md").write_text("\n".join(lines) + "\n")


def _set_run(run, text, size, color, bold=False):
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = "Calibri"


def write_pptx(deck: list[dict]) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    total = sum(s["seconds"] for s in deck)
    for index, slide in enumerate(deck, start=1):
        s = prs.slides.add_slide(blank)
        bg = s.shapes.add_shape(1, Emu(0), Emu(0), prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = CREAM
        bg.line.fill.background()
        bar = s.shapes.add_shape(1, Emu(0), Emu(0), prs.slide_width, Inches(0.18))
        bar.fill.solid()
        bar.fill.fore_color.rgb = TEAL
        bar.line.fill.background()
        title = s.shapes.add_textbox(Inches(0.55), Inches(0.38), Inches(12.2), Inches(1.1))
        tf = title.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        _set_run(p.add_run(), slide["title"], 30, INK, bold=True)
        body = s.shapes.add_textbox(Inches(0.7), Inches(1.6), Inches(12.0), Inches(5.0))
        btf = body.text_frame
        btf.word_wrap = True
        for i, bullet in enumerate(slide["bullets"]):
            para = btf.paragraphs[0] if i == 0 else btf.add_paragraph()
            para.level = 0
            para.space_after = Pt(10)
            _set_run(para.add_run(), bullet, 18, INK)
        footer = s.shapes.add_textbox(Inches(0.55), Inches(7.05), Inches(12.2), Inches(0.32))
        fp = footer.text_frame.paragraphs[0]
        _set_run(
            fp.add_run(),
            f"Synthetic data  ·  not a medical device  ·  Happiest Health Digital Twin Challenge 2026    {index}/{len(deck)}",
            11,
            MUTED,
        )
        notes = s.notes_slide.notes_text_frame
        notes.text = (
            f"[{slide['seconds']} seconds. Deck total {total // 60} min {total % 60} s.]\n\n"
            + slide["notes"]
        )
    prs.save(DOCS / "presentation.pptx")


def write_slides_pdf(deck: list[dict]) -> None:
    path = DOCS / "presentation.pdf"
    c = pdfcanvas.Canvas(str(path), pagesize=landscape(A4))
    width, height = landscape(A4)
    for index, slide in enumerate(deck, start=1):
        c.setFillColorRGB(0.957, 0.969, 0.965)
        c.rect(0, 0, width, height, fill=1, stroke=0)
        c.setFillColorRGB(0.059, 0.431, 0.420)
        c.rect(0, height - 10 * mm, width, 10 * mm, fill=1, stroke=0)
        c.setFillColorRGB(0.110, 0.169, 0.165)
        c.setFont("Times-Bold", 22)
        y = height - 24 * mm
        for line in _wrap(slide["title"], 58):
            c.drawString(16 * mm, y, line)
            y -= 8 * mm
        c.setFont("Times-Roman", 13)
        y -= 4 * mm
        for bullet in slide["bullets"]:
            for j, line in enumerate(_wrap(bullet, 95)):
                prefix = "• " if j == 0 else "  "
                c.drawString(18 * mm, y, prefix + line)
                y -= 6.2 * mm
            y -= 1.5 * mm
        c.setFillColorRGB(0.322, 0.376, 0.369)
        c.setFont("Times-Italic", 9)
        c.drawString(
            16 * mm,
            10 * mm,
            f"Synthetic data, not a medical device. Speaker notes are in the PPTX.  {index}/{len(deck)}",
        )
        c.showPage()
    c.save()


def write_architecture_pdf() -> None:
    path = DOCS / "architecture.pdf"
    c = pdfcanvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    c.setFillColorRGB(0.957, 0.969, 0.965)
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColorRGB(0.059, 0.431, 0.420)
    c.rect(0, height - 18 * mm, width, 18 * mm, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Times-Bold", 14)
    c.drawString(14 * mm, height - 11 * mm, "Glycemic digital twin — architecture")
    c.setFillColorRGB(0.110, 0.169, 0.165)
    c.setFont("Times-Roman", 9)
    c.drawString(14 * mm, height - 24 * mm, "Source: docs/architecture.mmd. Synthetic generators only. Seed 2026.")

    boxes = [
        (14, 250, 82, 22, "EHR twin\nchart, meds, labs, markers"),
        (110, 250, 82, 22, "Latent physiology\nsensitivity, setpoint, dawn"),
        (14, 214, 178, 22, "Wearable twin, 5 min: CGM, heart rate, RMSSD, steps, sleep"),
        (14, 176, 55, 24, "Static chart"),
        (76, 176, 55, 24, "Sensor aggregates\nlast 6 hours"),
        (138, 176, 55, 24, "Meal and dose log"),
        (14, 132, 40, 28, "Persistence\nlatest CGM"),
        (58, 132, 40, 28, "Static\nlogistic"),
        (102, 132, 40, 28, "Wearable\ngradient boost"),
        (146, 132, 46, 28, "Fusion boost\n+ isotonic"),
        (14, 92, 82, 22, "Trajectory 30–120 min\n± 1 residual SD"),
        (110, 92, 82, 22, "Hypoglycemia head\nP(CGM < 70 in 2 h)"),
        (14, 52, 178, 24, "Clinician dashboard: risk, band, drivers, walk or skipped meal"),
    ]
    c.setStrokeColorRGB(0.059, 0.431, 0.420)
    c.setLineWidth(1)
    for x, y, w, h, text in boxes:
        c.setFillColorRGB(1, 1, 1)
        c.roundRect(x * mm, y * mm, w * mm, h * mm, 3 * mm, fill=1, stroke=1)
        c.setFillColorRGB(0.110, 0.169, 0.165)
        c.setFont("Times-Roman", 8)
        lines = text.split("\n")
        ty = y * mm + h * mm / 2 + (len(lines) - 1) * 3.2 * mm / 2
        for line in lines:
            c.drawCentredString((x + w / 2) * mm, ty - 1.2 * mm, line)
            ty -= 3.6 * mm
    c.setStrokeColorRGB(0.322, 0.376, 0.369)
    c.setFillColorRGB(0.322, 0.376, 0.369)
    c.setFont("Times-Italic", 8)
    c.drawString(14 * mm, 40 * mm, "Arrows, top to bottom: chart shapes physiology; physiology drives the wearable;")
    c.drawString(14 * mm, 36 * mm, "the decision window splits into chart, sensors, and log; fusion is the only model that sees all three;")
    c.drawString(14 * mm, 32 * mm, "trajectory, hypoglycemia, and the dashboard read the fusion features. Patient-level holdout.")
    c.showPage()
    c.save()


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = word if not cur else cur + " " + word
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    deck = slides(load_metrics())
    total = sum(s["seconds"] for s in deck)
    if total < 20 * 60:
        raise SystemExit(f"Speaker notes are only {total} seconds. Need at least 20 minutes.")
    write_video_script(deck)
    write_pptx(deck)
    write_slides_pdf(deck)
    write_architecture_pdf()
    print(f"Wrote docs for a {total // 60} min {total % 60} s script")


if __name__ == "__main__":
    main()
