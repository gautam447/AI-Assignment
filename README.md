# Glycemic Digital Twin for Type 2 Diabetes

**Happiest Health Digital Twin Challenge 2026**

A clinician-facing proof of concept. It estimates the probability that continuous glucose will reach **≥180 mg/dL within the next 2 hours**, fusing a static outpatient chart with a 6-hour wearable window. A second head estimates the probability of **glucose <70 mg/dL** in that same horizon. Every training row and every person on the dashboard is synthetic.

This is not a medical device and it is not for dosing.

## Team

Fill these in before you submit. Do not leave them blank, and do not invent a person.

| Field | Value |
| --- | --- |
| Team name | FILL IN |
| College | FILL IN |
| Team leader | FILL IN |
| Phone | FILL IN |
| Email | FILL IN |

## Problem statement

People with type 2 diabetes are harmed by glucose that spikes after meals and, less often, by glucose that falls too far after a medicine. A clinic HbA1c says how the last three months went. It does not say whether *this* lunch, on *this* medicine, is about to cross 180 mg/dL. A CGM trace says where glucose is now. It does not know that a sulfonylurea was taken, that insulin is on board, or that a large rice meal is logged for the next hour.

The twin asks a timed question a doctor can act on in a visit or a remote check-in:

> In the next 2 hours, will CGM glucose reach at least 180 mg/dL? Separately, might it fall below 70 mg/dL?

The primary output is a calibrated probability, a short trajectory with a residual band, the drivers of that probability, and one counterfactual (a 15-minute brisk walk, or skipping the next logged meal).

## India healthcare use case

The ICMR–INDIAB study reported about 101 million people living with diabetes in India and about 136 million with prediabetes (Anjana et al., *Lancet Diabetes & Endocrinology*, 2023). Many meals are built around rice or wheat, so the post-meal rise is a daily event, not an edge case. Follow-up is often an HbA1c every few months. CGM is still mostly paid for out of pocket and clustered in larger cities.

The intended setting is a district-hospital diabetes clinic or a tele-endocrinology review, not an intensive-care ward. The chart already holds age, BMI, HbA1c, fasting glucose, lipids, eGFR, and medicines (metformin, a sulfonylurea, insulin). A 6-hour wearable window adds CGM, heart rate, RMSSD, steps, and sleep. The output is an alert the clinician can put next to advice they already give: walk after the meal, do not skip the meal if a sulfonylurea or insulin was taken, or bring the person back sooner if the chart and the trace disagree.

Units follow Indian laboratory practice: glucose and lipids in mg/dL, HbA1c in percent, eGFR in mL/min/1.73 m². BMI excess is measured from 23 kg/m², the WHO Asian action point.

## What the numbers below are

They are from one seeded run of this repository: **320 synthetic patients, 3 simulated days, patient-level 70/15/15 split, seed 2026**. Test set: 48 patients, 6,144 decision windows, hyperglycemia prevalence **0.3185**. Nothing here is a literature benchmark, and nothing was typed in by hand. The file is `artifacts/metrics.json`. Rebuild with `python -m twin.train` if you change the generator; then regenerate the slides with `python -m twin.build_docs`.

| Model | What it sees | AUROC | AUPRC | Sensitivity | Specificity achieved | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| Persistence logistic | Latest CGM value only | 0.8355 | 0.7910 | 0.7251 | 0.8073 | 0.0241 |
| Static logistic regression | EHR chart only, no clock | 0.7618 | 0.5968 | 0.5764 | 0.8108 | 0.0375 |
| Wearable gradient boosting | 6-hour sensor aggregates only | 0.9477 | 0.9010 | 0.9090 | 0.8106 | 0.0102 |
| **Fusion gradient boosting** | Chart + sensors + meal and dose log | **0.9782** | **0.9535** | **0.9852** | **0.8214** | **0.0125** |

Sensitivity is reported at the most sensitive threshold whose specificity is at least 0.80 on the test set. The threshold itself was chosen on the validation patients. Fusion’s validation threshold is 0.0979.

Fusion also leads on the naturalistic slice that drops any window where a walk or a skipped meal was logged (5,936 windows): AUROC 0.9781 versus 0.9499 wearable, 0.8366 persistence, and 0.7588 static. The lift is not an artifact of the action flags.

**Lead time and alert usefulness (fusion, test patients).** Of 314 usable rises to 180 mg/dL, the model alerted before the crossing on 98.73%, with a median lead of **100 minutes**. 92.58% of detected episodes had at least 30 minutes of lead. Raw alerts, counted every 30 minutes while glucose is still under 180, fire **11.89 times per patient-day** with a positive predictive value of **0.5759**. Collapsing consecutive alerts into episodes gives **2.90 new alert episodes per patient-day**. The long lead is real in this generator because the meal log is known before glucose moves. A person who never logs a meal would look closer to the wearable row.

**Hypoglycemia, secondary head.** Test prevalence is only 0.0127, so AUROC flatters the model. Fusion hypoglycemia AUROC 0.9636, AUPRC **0.6606** (against a 0.0127 baseline rate), sensitivity 0.9487 at achieved specificity 0.9278, ECE 0.0039. The validation threshold is 0.0235. Read the probability, not only the badge. AUPRC would be the first number to improve with more low-glucose events or a less formulaic mismatch between dose and meal.

**Trajectory.** Mean absolute error of the fusion regressors on the test set: 4.11 mg/dL at 30 min, 7.18 at 60, 9.17 at 90, 9.81 at 120. The ±1 residual-SD band covered 79–86% of test points. Those errors are tight because the synthetic meal log and the glucose equation share a mechanism. Real CGM would be worse. That is a limitation, not a claim of clinical accuracy.

**Where the lift comes from.** Group permutation on the test set drops fusion AUROC by 0.246 when recent glucose is shuffled and by 0.183 when the meal and insulin log is shuffled. Shuffling HbA1c, BMI, and duration drops AUROC by only 0.012. The chart-only model can rank who lives high (AUROC 0.76) but cannot time the next two hours. Once the recent trace and the meal log are in hand, the labs add a smaller slice. That is the honest reading.

**Counterfactuals.** On 1,487 rising test windows, forcing the “skip next meal” flag and clearing planned carbs lowers mean fusion probability by **0.106**. Forcing the walk flag alone, without changing steps, lowers it by only **0.017**. The dashboard walk resimulates the steps, so the sensor window changes and the probability moves with it. The meal skip is the action the classifier has actually learned as a flag.

## Technical stack and model

| Piece | Choice |
| --- | --- |
| Language | Python 3 |
| Tables and arrays | pandas 2.2.3, numpy 1.26.4 |
| Models | scikit-learn 1.5.2 |
| Dashboard | Streamlit 1.44.1, Plotly 5.24.1 |
| Slides | python-pptx 1.0.2 and reportlab 4.2.5 |
| Artifact | `artifacts/model.joblib` |

Static baseline: `StandardScaler` + L2 logistic regression on the chart (age, sex, BMI, diabetes duration, HbA1c, fasting glucose, LDL, HDL, triglycerides, eGFR, systolic pressure, family history, simulated TCF7L2 and CDKAL1 allele counts, metformin, sulfonylurea, insulin dose, rice-forward diet, urban/rural).

Persistence: logistic regression on the latest CGM value.

Wearable model and fusion model: `HistGradientBoostingClassifier` (depth 6, learning rate 0.08, L2 0.15, early stopping), then isotonic regression fit on the validation patients only. The wearable model sees 6-hour aggregates: level, spread, 1-hour and 3-hour slope, acceleration, time above 140 and 180 and below 70, heart rate, RMSSD, steps, sleep fraction, deep-sleep fraction, and time of day. Fusion adds the chart and the log: minutes since the last meal, carbohydrates in the last 2 hours, carbohydrates planned in the next 2 hours, minutes since insulin and the recent dose, minutes since a sulfonylurea, and the two action flags.

Trajectory: four gradient-boosting regressors for glucose at +30, +60, +90, and +120 minutes. The band is ±1 residual standard deviation from validation.

Logged walks and skipped meals are upweighted 3× while fusion and the hypoglycemia head are fit, so those rarer rows are not ignored. Test metrics are not reweighted.

The wearable series is not independent noise. A latent physiology (insulin sensitivity, between-meal setpoint, dawn amplitude, meal size) is a function of the chart. Meals, dawn, short sleep, steps, metformin, sulfonylurea, and insulin then move one glucose state. CGM is that state plus autocorrelated sensor noise. Labels use the CGM, not the hidden state.

## How to run

From a fresh clone, one command:

```bash
python run_demo.py
```

That installs `requirements.txt` if the imports are missing, loads the committed model (or trains one: 320 patients × 3 days, seed 2026), loads the 12-person demo cohort, and serves the dashboard at <http://localhost:8501>.

Other entry points:

```bash
python -m pytest                 # generators, a tiny train, and the saved comparison
python -m twin.train             # rebuild artifacts/model.joblib and metrics.json
python -m twin.build_docs        # rewrite the slides, PDFs, and video script from metrics.json
python run_demo.py --check       # print one probability and the saved fusion metrics
```

Click a row in the patient list. The four probabilities are the four models at that minute. The orange band is the fusion trajectory. The dotted line is the simulated future, which was not a model input. Choose “15-minute brisk walk now” or “Skip the next meal” and both the probability and the forward curve update.

`legacy/cifar-coursework/` is an older image-classification exercise. It is not part of this submission.

## Architecture

- Diagram (PDF): [docs/architecture.pdf](docs/architecture.pdf)
- Source (Mermaid): [docs/architecture.mmd](docs/architecture.mmd)

## Presentation

- Slides (PPTX, with timed speaker notes): [docs/presentation.pptx](docs/presentation.pptx)
- Slides (PDF): [docs/presentation.pdf](docs/presentation.pdf)

The deck covers the problem, the method, the synthetic data, the results from this run, a dashboard walkthrough, limitations, and clinical impact.

## Demo video

A timed script for an unlisted YouTube recording of **at least 20 minutes** is in [docs/video-script.md](docs/video-script.md). The speaker notes in the PPTX match that script.

No video URL is listed here. Record the walkthrough, set it to unlisted, and paste the URL into this section before you submit the form. Do not claim a link exists until the upload does.

## License

MIT. See [LICENSE](LICENSE). The copyright line is `FILL IN Team Name` until you replace it.

## Before you submit

1. Replace every **FILL IN** field: team name, college, leader name, phone, and email. Those fields are placeholders on purpose.
2. Rename the submission folder to `TeamName_CollegeName`.
3. Make this GitHub repository **public**.
4. Record the unlisted YouTube video from `docs/video-script.md` and add the URL above.
5. Submit the leader’s name, phone, email, and the public repository link.

Until those are done, the prototype runs, but the form is incomplete.
