# Recording script: glycemic digital twin

Unlisted YouTube walkthrough for the Happiest Health Digital Twin Challenge 2026.
No video has been uploaded. Paste the URL into the README only after it exists.

Timed length of the notes below: **35 minutes 10 seconds** if each section is delivered in its allotted time at a measured pace, about 140 words per minute. The speaker notes in `docs/presentation.pptx` are the same text.

Record the screen with the dashboard already running (`python run_demo.py`). Slides first, then the live panel, then the limitations slide. Do not read a team name, college, phone, or email that is still marked FILL IN.

| Start | Section |
| --- | --- |
| 00:00 | 1. A glycemic digital twin for the next two hours (70 s) |
| 01:10 | 2. Why this question, in India (140 s) |
| 03:30 | 3. The decision the twin is allowed to make (110 s) |
| 05:20 | 4. What this prototype is not (90 s) |
| 06:50 | 5. Synthetic chart: an Indian outpatient file (120 s) |
| 08:50 | 6. The wearable stream is coupled to that chart (150 s) |
| 11:20 | 7. Train, validation, test: by patient (100 s) |
| 13:00 | 8. Four models, same label (120 s) |
| 15:00 | 9. Held-out comparison (150 s) |
| 17:30 | 10. The lift survives without the action flags (130 s) |
| 19:40 | 11. Lead time, and how often it would buzz (140 s) |
| 22:00 | 12. Trajectory band, and the low-glucose head (120 s) |
| 24:00 | 13. Dashboard: what to click (180 s) |
| 27:00 | 14. One counterfactual, computed in the open (120 s) |
| 29:00 | 15. Limitations, in the order they matter (140 s) |
| 31:20 | 16. What would change practice, if the signal survived real data (90 s) |
| 32:50 | 17. Reproduce the run (70 s) |
| 34:00 | 18. Before this is a submission (70 s) |
| 35:10 | End |

## 00:00  Slide 1 — A glycemic digital twin for the next two hours

*On screen: 70 seconds.*

- Happiest Health Digital Twin Challenge 2026
- Type 2 diabetes, Indian clinic units (mg/dL)
- Synthetic patients only. Not a medical device.
- Team, college, leader, phone, email: FILL IN

Introduce the project in one sentence. We estimate the probability that CGM glucose reaches at least 180 milligrams per decilitre within two hours, and we raise a second alert if it may fall below 70. The setting is type 2 diabetes in India. Everything you will see was generated. There is no real patient in the training set and no real patient on the dashboard. Say the team name, college, and leader only after the FILL IN fields in the README have been replaced. Do not invent them on the recording. Tell the audience the recording follows the dashboard as a clinician would use it, then the held-out numbers from seed 2026.

## 01:10  Slide 2 — Why this question, in India

*On screen: 140 seconds.*

- ICMR–INDIAB: about 101 million people with diabetes, about 136 million with prediabetes (Anjana et al., Lancet Diabetes & Endocrinology, 2023)
- Rice- and wheat-based meals make the post-meal rise a daily event
- Follow-up is often an HbA1c every few months; CGM is still mostly out of pocket
- A district clinic or tele-endocrinology review needs a 2-hour answer, not another 3-month average

Ground the use case. Cite ICMR–INDIAB as published epidemiology, not as a number this project measured. The practical gap is timing. HbA1c tells you the quarter. It does not tell you whether today's lunch, on today's sulfonylurea or insulin, is about to cross 180. CGM tells you where glucose is now. It does not know the dose that was taken twenty minutes ago, and it does not know that a large meal is written on the log for the next hour. Describe a district-hospital diabetes clinic or a remote review: short appointment, chart already in the file, sometimes a few hours of wearable data. The product has to be readable in that visit. Mention units explicitly: milligrams per decilitre, HbA1c in percent, eGFR in mL per minute per 1.73 square metres, and BMI excess measured from 23, the WHO Asian action point, not from 25.

## 03:30  Slide 3 — The decision the twin is allowed to make

*On screen: 110 seconds.*

- Primary label: CGM maximum in the next 2 hours ≥ 180 mg/dL
- Secondary label: CGM minimum in the next 2 hours < 70 mg/dL
- Inputs: the static chart, the last 6 hours of sensors, the meal and dose log
- Outputs: calibrated probability, a 4-point trajectory with a residual band, drivers, one counterfactual

Be precise about the label so nobody thinks this is a generic 'risk score'. At a decision minute, the model may use glucose, heart rate, RMSSD, steps, and sleep from the previous six hours, the chart, and what the log already says about meals and doses. It may not use any future glucose. The positive class is whether the CGM reading, including sensor noise, reaches 180 or more at any 5-minute sample in the next two hours. The secondary class is a fall below 70. We also predict the level at 30, 60, 90, and 120 minutes. Say clearly that 180 and 70 are operational thresholds for this prototype, aligned with common hyperglycemia and level-1 hypoglycemia cut-points, not a personalised target.

## 05:20  Slide 4 — What this prototype is not

*On screen: 90 seconds.*

- Not trained on any real electronic health record or CGM export
- Not a dosing calculator and not a closed-loop controller
- Genetic fields are simulated allele counts, not sequenced genotypes
- A tight error on synthetic glucose is not clinical accuracy

Spend time here so the rest of the talk is credible. Judges have seen projects that blur synthetic and real. Say the generator, the split, the metrics file, and the dashboard all use the same seeded simulator. If a number appears on a slide, it was written by python -m twin.build_docs from artifacts/metrics.json after python -m twin.train. We did not pick a flattering subset after looking at the test labels. The patient split is by person, 70, 15, 15, so no window from a test patient is in the fit. Also say what would make this a different project: a real CGM study with a locked protocol, a site in India, and a calibration check on that site's devices.

## 06:50  Slide 5 — Synthetic chart: an Indian outpatient file

*On screen: 120 seconds.*

- 320 people for the reported run, seed 2026. Demo panel uses a different seed.
- Age, sex, city and state, BMI from a distribution centred near 27
- HbA1c mixture: most people near a treated target, a smaller group well above it
- Medicines: metformin, sulfonylurea, insulin total daily dose. Labs in mg/dL.
- Simulated TCF7L2 and CDKAL1 allele counts and a family-history flag

Walk through one row as if you were opening a file. Name the city as a generated label, not a claim that a real clinic sent data. Explain the medicine mix: metformin is common, sulfonylureas remain common where cost shapes the formulary, and insulin is used both in poorly controlled diabetes and in some people sitting nearer target, because those are the people who can actually go low if a meal is missed. The allele counts are binomial draws at fixed frequencies. They change insulin sensitivity inside the simulator. They are not a genetic test. Fasting glucose, lipids, and eGFR are generated to move with HbA1c, age, and duration so the chart is internally consistent. The same seed rebuilds the same cohort.

## 08:50  Slide 6 — The wearable stream is coupled to that chart

*On screen: 150 seconds.*

- 5-minute CGM, heart rate, RMSSD, steps, sleep stage
- One glucose state: meals, dawn, short sleep, walking, metformin, sulfonylurea, insulin
- A logged meal raises glucose later if sensitivity is low; insulin without that meal can fall below 70
- CGM is the state plus autocorrelated noise. Labels use the CGM.

This is the part to slow down. Independent noise streams would let a fusion model cheat or, worse, learn nothing. Here the chart sets a between-meal setpoint, a dawn rise, and how large an 80 gram carbohydrate meal is allowed to be. Rice-forward diets increase lunch carbohydrate. A walk adds steps and glucose disposal, larger if insulin or a sulfonylurea is on board. Insulin and sulfonylurea lower the target after a delay of about half an hour, so a decision taken just after a dose does not yet show the fall on the CGM. If the paired meal is missing, an extra disposal can take a near-target person below 70. A person with HbA1c around 11 often stays high instead of going low. That matches the generator and it is why hypoglycemia is uncommon in the test set. Heart rate rises with steps and meals and falls in sleep. RMSSD falls as glucose rises and is higher in deep sleep. Sleep debt the next morning lifts the setpoint. Show, if you have the tests open, that the same schedule with a higher HbA1c has a higher mean glucose, that insulin lowers the lunch peak, and that a forced walk lowers it further without changing the history before the walk.

## 11:20  Slide 7 — Train, validation, test: by patient

*On screen: 100 seconds.*

- Seed 2026: 320 patients × 3 days
- Windows — train 28672, validation 6144, test 6144
- Patients — 224 / 48 / 48
- Test prevalence of a 2-hour rise to ≥180: 0.3185
- Decision every 30 minutes. Isotonic calibrators see validation rows only.

Read the counts out loud so the split is concrete. Three hundred and twenty synthetic people, three days each, seed 2026. The test set is 48 people and 6144 windows. About 31.9% of those windows reach 180 in the next two hours. That is common enough that accuracy would be a vanity metric, which is why the table uses AUROC, AUPRC, sensitivity at a fixed specificity, and calibration. The threshold for an alert is chosen on validation to sit at specificity at least 0.80, then frozen and applied to test. We report the specificity we actually got, which is not always exactly 0.80 because scores are discrete. Early stopping inside gradient boosting uses a slice of the training patients only. It does not peek at the test people.

## 13:00  Slide 8 — Four models, same label

*On screen: 120 seconds.*

- Persistence: logistic regression on the latest CGM value
- Static: L2 logistic regression on the chart, no clock
- Wearable: histogram gradient boosting on 6-hour sensor aggregates
- Fusion: the same booster on chart + sensors + meal and dose log, then isotonic calibration
- Logged walks and skipped meals are upweighted 3× in the fusion fit. Test metrics are not.

Explain why these baselines, not a leaderboard of fashionable networks. Persistence is the right dumb answer: if you are already at 210 you will probably still be high, and if you are at 110 and flat you probably will not cross 180. Any serious model has to beat that. The static logistic is the chart a doctor already has, with no knowledge of what this afternoon looks like. The wearable model is the sensor vendor's model: slope, level, steps, heart rate, sleep, time of day, and no medicines. Fusion is the same tree booster as the wearable model, so a win is not an algorithm trick. The extra inputs are the chart and the log: planned carbohydrates, recent carbohydrates, minutes since insulin, the recent dose, minutes since a sulfonylurea, and two action flags. Say the 3× weight in one sentence and immediately say the test set was not reweighted. Trajectory heads are four regressors on the fusion features. The band is plus or minus one validation residual standard deviation, not a conformal interval.

## 15:00  Slide 9 — Held-out comparison

*On screen: 150 seconds.*

- Persistence — AUROC 0.8355, AUPRC 0.7910, sensitivity 0.7251 at specificity 0.8073
- Static logistic — AUROC 0.7618, AUPRC 0.5968, sensitivity 0.5764 at specificity 0.8108
- Wearable — AUROC 0.9477, AUPRC 0.9010, sensitivity 0.9090 at specificity 0.8106
- Fusion — AUROC 0.9782, AUPRC 0.9535, sensitivity 0.9852 at specificity 0.8214, ECE 0.0125

Read the four AUROCs slowly. Fusion is 0.9782, wearable 0.9477, persistence 0.8355, static 0.7618. Fusion is ahead of all three. The gap over the wearable model is 0.0305 AUROC and 0.0525 AUPRC. The gap over persistence is larger, 0.1427 AUROC, which is what you want if the latest glucose value is already a strong score. Do not hide the static result. An AUROC of 0.7618 means the chart ranks who tends to go high. It does not time the day. Sensitivity at specificity around 0.80 is 0.9852 for fusion, 0.9090 for wearable, 0.7251 for persistence, and 0.5764 for the chart. If you only remember one comparison, remember that fusion wins on both ranking and on the high-specificity operating point, on people it did not train on.

## 17:30  Slide 10 — The lift survives without the action flags

*On screen: 130 seconds.*

- Naturalistic test windows (no logged walk, no logged skip): 5936
- Fusion AUROC 0.9781 vs wearable 0.9499, persistence 0.8366, static 0.7588
- Permutation: recent glucose drops AUROC by 0.246; meal and insulin log by 0.183
- HbA1c, BMI, and duration together drop AUROC by only 0.012
- Fusion ECE 0.0125. Static ECE 0.0375.

This slide is the defence against a fair objection: maybe fusion only wins because you told it that someone was about to walk or skip a meal. On the 5936 test windows where both action flags are off, fusion AUROC is still 0.9781 and the wearable model is 0.9499. The permutation test says what the model actually uses. Shuffling the recent glucose level costs about 0.246 AUROC. Shuffling the meal and insulin log costs about 0.183. Shuffling HbA1c, body size, and duration costs about 0.012. So the fusion advantage over a sensor model is mostly the log, not a magical reading of the HbA1c. The chart still matters for the static baseline, and it is in the fusion feature vector, but once the trace and the log are present its marginal contribution is small. Say that plainly. It is a better scientific claim than pretending every lab drives the score. Calibration: fusion expected calibration error is 0.0125 after isotonic regression on validation. That is acceptable on this synthetic test and it is not a guarantee on a new device or a new city.

## 19:40  Slide 11 — Lead time, and how often it would buzz

*On screen: 140 seconds.*

- Usable episodes on test patients: 314. Detected before the crossing: 98.7%.
- Median lead 100 min. Share with at least 30 min lead: 92.6%.
- Alerts while still under 180: 11.89 per patient-day. PPV 0.5759.
- New alert episodes, collapsing consecutive alarms: 2.90 per patient-day.

Define an episode before you quote the lead. An onset is the first sample at or above 180 after being below it. We ignore an onset that has no scored decision in the prior two hours. An alert counts only if glucose is still under 180 at the decision, so we are not taking credit for announcing a high that has already happened. Median lead is 100 minutes. 92.6% of detected episodes had half an hour or more. That lead is long because the simulator's meal log is known before the glucose moves. If the person does not log the meal, you should expect something closer to the wearable model, which has to wait until the slope is visible. Say that limitation in the same breath as the number. The annoying number is 11.9 raw alerts per patient-day. The score is evaluated every 30 minutes, and many people stay in a high-risk region through a whole meal. Collapsing consecutive alerts gives 2.9 episodes per patient-day. Positive predictive value of a raw alert is 0.5759. A clinic would alert on the rising edge and would retune the threshold for its own cost of a false alarm. We froze specificity 0.80 instead of shopping for a prettier operating point.

## 22:00  Slide 12 — Trajectory band, and the low-glucose head

*On screen: 120 seconds.*

- MAE: 4.1 mg/dL at 30 min, 7.2 at 60, 9.2 at 90, 9.8 at 120
- ±1 residual-SD coverage on test: 79.4% to 82.4%
- Hypoglycemia prevalence 0.0127. AUROC 0.9636. AUPRC 0.6606.
- Hypoglycemia sensitivity 0.9487 at specificity 0.9278. ECE 0.0039.

Two different levels of honesty. The trajectory errors are small: about 4 milligrams at half an hour and about 10 at two hours. Coverage of the one-standard-deviation band is roughly 80 to 86 percent, wider than a Gaussian 68 percent because the residuals are not perfectly normal. Do not sell this as sensor-grade forecasting. The meal appearance kernel and the model see the same planned carbohydrates. A real stomach is less obedient. The right next experiment is to hide the planned carbs and see how fast the MAE degrades. Hypoglycemia is the weaker clinical claim. Only 1.3% of test windows go below 70, because many people on insulin in this generator live too high to reach a biochemical low even if they miss a meal. AUROC 0.9636 looks strong. AUPRC 0.6606 is the number to remember: better than the 0.0127 base rate, not a solved detection problem. The dashboard still shows the low-glucose probability because a sulfonylurea or insulin with no meal is the scenario a clinician is afraid of, and the generator does produce it.

## 24:00  Slide 13 — Dashboard: what to click

*On screen: 180 seconds.*

- Open with python run_demo.py. Twelve synthetic people, ranked by fusion probability.
- Select a row. Compare persistence, static, wearable, and fusion at the same minute.
- Teal line: history the model may use. Dotted line: simulated future it did not see.
- Orange band: fusion trajectory ± 1 residual SD. 180 and 70 are marked.
- Drivers: change in probability when a group is replaced by the training median.

This is the live section. Start the app before you record, with python run_demo.py, and widen the window. Read the banner: synthetic India cohort, seed 2026. Click the top row, then a middle row around a fifty percent probability, then a row whose hypoglycemia probability is high and hyperglycemia probability is near zero. On each person, point at the four metrics. The point of the middle row is that the models disagree: persistence and the wearable model are looking at a trace that has not risen yet, while fusion can see a meal on the log. If they do not disagree on the person you clicked, say what you actually see. Do not invent a disagreement. Trace the chart with the cursor. History is teal. The dotted continuation is the simulator's future, drawn so a judge can see whether the alert was early. It is not an input. The orange markers are 30, 60, 90, and 120 minutes. Open the driver chart. A bar to the right means this person's values raise the probability relative to a typical training window. Recent glucose and the meal log should usually dominate. If a genetic marker is near zero, say so. The permutation test already told us the labs are a small marginal effect. Mention the city, the HbA1c, and the medicine line so the file feels like a chart, and remind the viewer the name is generated.

## 27:00  Slide 14 — One counterfactual, computed in the open

*On screen: 120 seconds.*

- Skip the next meal: the log flag is cleared and planned carbohydrates go to zero.
- On 1487 rising test windows, that lowers mean fusion probability by 0.106.
- A 15-minute walk resimulates steps. The flag alone, without new steps, moves mean risk by only 0.017.
- The blue dashed curve is the same physiology with the action applied. History before the decision matches.

Do the click. Pick a person who still has a meal on the log in the next two hours. Switch on Skip the next meal. Read the probability before and after from the screen, not from memory. The test-set average drop is 0.106 on rising windows. A single person can move more or less. Then switch to the 15-minute walk. Be honest about the mechanism. Flipping the walk flag and leaving the sensor window untouched changes mean risk by only 0.017. The trees barely split on that rare flag. The dashboard instead resimulates the walk, so steps in the last 30 minutes jump and the glucose equation removes on the order of a few tens of milligrams per decilitre. The probability moves because the sensor features move. The blue curve is that resimulation. Say that the synthetic walk effect was made large enough to see on a five-minute grid. Published walks after meals are often smaller and noisier. If the selected person has no meal coming, the skip control should barely move. Say that too. A counterfactual that does nothing is still a result.

## 29:00  Slide 15 — Limitations, in the order they matter

*On screen: 140 seconds.*

- Synthetic physiology. Real CGM, real logging, and real missed doses will be messier.
- Meal-log lead time will shrink the moment people do not log.
- Hypoglycemia AUPRC is 0.66 on a 1.3% event. AUROC alone is the wrong headline.
- Static labs add little once the trace and the log are present.
- No site, no device, no clinician in the loop, no fairness audit across states.

Do not rush the limitations. They are part of the result. First, the generator is a discrete-time equation, not a fitted physiological model. We did not force week-long mean glucose to equal the ADAG average implied by HbA1c. The setpoint is the between-meal target; meals sit on top; walks and missed meals pull weeks around. Correlation with HbA1c is there. Identity with a textbook formula is not. Second, planned carbohydrates are visible to fusion and not to the wearable baseline. That is a product choice, and it is also the main reason the median lead is 100 minutes. Third, trajectory MAE under 10 milligrams at two hours will not survive contact with unlogged meals. Fourth, the low-glucose head needs more events before anyone should pick an alert threshold for a ward. Fifth, upweighting action rows 3× changes the fit; we reported unweighted test metrics, but a different weight would move the small walk coefficient. Sixth, cities in the demo are labels for a story, not a sample of Indian diabetes. We did not test whether error differs by age, sex, or urban versus rural. Seventh, there is no prospective silent trial and no human factors test of the alert rate.

## 31:20  Slide 16 — What would change practice, if the signal survived real data

*On screen: 90 seconds.*

- Put the 2-hour probability next to the HbA1c in a district clinic review
- Use the meal-skip counterfactual as a safety check when a sulfonylurea or insulin was taken
- Use the walk counterfactual as a prescription the person can finish before the peak
- Retune the threshold locally. Specificity 0.80 was a reporting choice, not a guideline.

Keep this conditional. If, on real CGM with real logs, fusion still beats persistence and a sensor-only model by a margin a clinic can feel, the workflow is modest. The doctor already knows the medicines. The new object is a probability for this afternoon and a button that shows the same model with the next meal removed or a short walk inserted. The safety use is the missed meal on a sulfonylurea or insulin, which is common advice already and which this prototype can illustrate. The post-meal walk is the other advice clinicians already give; the screen makes the size of the change visible. None of that replaces hypoglycemia education, sick-day rules, or a human deciding a dose. A deployment path that respects the evidence would be silent scoring on a consented cohort, then a threshold workshop with the clinic, then a stepped introduction. We have not done those steps.

## 32:50  Slide 17 — Reproduce the run

*On screen: 70 seconds.*

- python run_demo.py
- python -m pytest
- python -m twin.train    # writes artifacts/model.joblib and metrics.json
- python -m twin.build_docs
- MIT license. Architecture PDF and Mermaid source are in docs/.

Show the repository root. One command, python run_demo.py, installs pinned dependencies if they are missing, loads the committed artifact, and serves the dashboard. If the artifact is absent it trains 320 by 3 days at seed 2026. That is the run these slides describe. Tests cover seed stability, the coupling from HbA1c, meals, insulin, and walks, the absence of future glucose in the features, a tiny in-memory train, and the saved comparison. Point at docs/architecture.pdf and the Mermaid source. Point at the PPTX, which carries these speaker notes, and the PDF of the same slides.

## 34:00  Slide 18 — Before this is a submission

*On screen: 70 seconds.*

- FILL IN team name, college, leader name, phone, and email
- Rename the folder to TeamName_CollegeName
- Make the repository public
- Record this script, unlisted, at least 20 minutes, and paste the URL into the README
- No video URL is claimed in the repository today

Close on the administrative gaps so you do not imply they are done. The README marks team name, college, leader, phone, and email as FILL IN. Leave them as FILL IN on the recording if they are still empty, and say the form cannot be sent until a real team replaces them. Rename the submission folder to TeamName_CollegeName. The GitHub repository must be public before the leader's name, phone, email, and repository link are submitted. This talk is the script for an unlisted YouTube video of at least twenty minutes. The repository does not contain a video URL. Add it only after the upload exists. Thank the judges, restate the one-sentence task, and stop.

