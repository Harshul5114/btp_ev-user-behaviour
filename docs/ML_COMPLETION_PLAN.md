# ML completion plan — EV user behaviour prediction

Updated 2026-11-26. This is the active project plan and supersedes the earlier
charging-policy-first priorities in `README_BTP_Five_Day_Plan.md`.

## Objective and research question

Produce reproducible, meaningful results answering:

**How well can past charging behaviour predict a user's next recorded charging
event, and how much does user history improve predictions over simple baselines?**

The primary contribution is behaviour prediction. Converting predictions into
charging demand is a downstream demonstration. The immediate-versus-23:00
comparison is supporting material; further charging optimization and grid
integration are deferred until the ML result and presentation are complete.

Meaningful does not mean guaranteeing that ML wins. A supported result about
where prediction works, where it fails and what history is needed is useful.

## Starting point

- Norway Dataset 1 contains about 35,377 residential charging reports from
  267 users at 12 sites; confirm the actual usable cohort for each target.
- Session-energy and timing pipelines already exist in `src/models/` and
  notebooks 06–07. Reuse and repair these before adding a model.
- Saved energy metrics show little MAE improvement over user-history medians
  and weaker unseen-user performance. Documentation and saved numbers disagree.
- Timing code exists, but claimed output artifacts need reproduction.
- A synthetic charging generator and synthetic checks already exist. Inspect
  them before extending them; do not build a second generator unnecessarily.
- NHTS travel EDA, trip chains and charging simulation are supporting assets.
  They are not longitudinal charging histories matched to Norway users.

## Stage 1 — Establish trustworthy experiments

Reproduce both existing ML scripts, fix required dependencies and write one
authoritative results table. Record source files, usable rows/users, exclusions,
prediction timestamp, split dates, configuration and seed. Generate the docs
from or reconcile them against the saved metrics; do not retain unverified claims.

Audit a few important causal constraints rather than building an extensive test
suite:

- A previous session's energy and duration become available at completion,
  not simply because its plug-in timestamp precedes the current prediction.
- A previous inter-session gap is known only when its next session begins.
- Training labels must be available before the training cutoff. Purge records
  whose completion/next-event label crosses a validation or test boundary.
- Fit imputation, scaling, site encodings, feature selection and model tuning
  inside training folds. Audit stage-one predictions and target encodings for
  temporal contamination; do not assume ordinary shuffled folds are causal.
- Sequential test-history updates are allowed only after observations become
  available. Report that protocol separately from a fixed-origin forecast.

**Deliverable:** corrected reproducible scripts and an audit note identifying
which earlier results are confirmed, changed or still unsupported.

## Stage 2 — Predict a clearly defined behaviour

Use existing targets with these explicit prediction points:

| Priority | Question | Prediction point | Target and interpretation |
|---|---|---|---|
| Primary | When will this user next connect at the recorded sites? | Current recorded plug-out | Hours until next recorded plug-in; recover the full timestamp |
| Secondary | When will the current connection end? | Current plug-in | Connection duration and recovered plug-out timestamp |
| Supporting | How much energy will this session deliver? | Current plug-in | Reported grid-delivered session kWh |

Connection duration is not active charging duration. The next recorded site
session is not necessarily the user's next charge anywhere. Do not interpret
these targets as driving departure or next-day travel prediction.

Inspect duplicates, overlapping sessions, negative gaps, long tails, observation
coverage and final sessions without an observed successor. Report exclusions
and censoring. Do not label a missing next session as zero charging or treat a
last-record gap as a completed duration. Results based on completed gaps must
state that conditioning and its potential bias.

If daily charge/no-charge prediction is considered later, first establish user
observation windows and trustworthy negative examples. Do not manufacture
non-charging days from missing records.

## Stage 3 — Baselines, evaluation and limited improvements

1. Compare training median, calendar-conditioned median, last observed user
   value and causal user-history median with the existing gradient boosting.
   Provide a training-only fallback for sparse history.
2. Use an earlier training block, a later validation block for decisions and a
   final chronological test block. Freeze the final test for reporting. Since
   earlier experiments have already inspected some held-out data, document
   that reuse and do not call the reused period an untouched test.
3. Separate users seen in training, users new at the chronological cutoff and
   a user-disjoint holdout. Distinguish zero-history performance from performance
   after new users accumulate observed history. Do not merge these claims.
4. Report MAE and RMSE in hours/kWh. For next plug-in, report full timestamp
   error and accuracy within useful windows (for example 6 and 24 hours).
   Circular clock-hour error alone can hide prediction on the wrong day.
5. For interval predictions, report nominal versus actual coverage, interval
   width and pinball loss. Report errors by history length, day type and site,
   with sample counts; show where rare long gaps dominate errors.
6. Run compact ablations: calendar/site only; plus causal history; optionally
   plus NHTS priors. Use Norway-only results as the main experiment. NHTS 2022
   priors applied to 2018–2021 Norway sessions are cross-population retrospective
   information, not a contemporaneously available operational forecast feature.
7. Make only validation-supported changes to history features, loss or tuning.
   Use a small search with the existing model family before adding complexity.
   Quantify paired MAE differences against the strongest baseline, with a
   user-grouped bootstrap interval where feasible; report user and session counts.

**Deliverable:** one readable comparison table, one timing-error plot, one
history-length/ablation plot and an honest conclusion per target. A model that
does not improve the baseline should remain a negative finding or use the
baseline downstream, not be advertised as a successful predictor.

## Stage 4 — Synthetic data where it adds evidence

Use synthetic data to investigate what can be learned under known assumptions,
not to replace weak real-data results with an easier benchmark.

- Reuse the existing generator for repeated users with commuter, daytime,
  irregular and multi-day charging patterns. Keep temporal order, event overlap
  constraints and completion-time feature availability explicit.
- Expose seeds, noise, history lengths, user mix, charger/battery assumptions
  and distribution-shift settings in configuration. Keep ground-truth archetypes
  separate from predictor inputs unless they would be observable in deployment.
- Start with a few controlled experiments: strong versus weak user regularity,
  short versus long user history, and new-user or shifted-schedule performance.
  Use separate seeds/users/time periods for training and evaluation and several
  generation seeds to check that conclusions do not depend on one draw.
- If calibrating synthetic distributions from real data, use training data only.
  Separate synthetic-only evaluation from any augmentation experiment.
- Try augmentation only if a specific limitation motivates it. Compare real-only
  training with real-plus-synthetic training on the same real held-out sessions.
  Keep it only if validation supports it; synthetic performance alone establishes
  neither generalization nor improved real-world prediction.
- Preserve raw data. Store generated records in `data/synthetic/` and results
  under separate synthetic output paths. Label every table and plot by provenance.

**Deliverable:** a compact controlled-study figure explaining when prediction
is learnable and how history/noise affect it. If real prediction remains weak,
present that limitation alongside the controlled result, without blending scores.

## Stage 5 — Connect behaviour prediction to demand, if time permits

Demonstrate the effect of prediction errors on a clearly bounded load outcome.
First compare recorded versus predicted connected-vehicle counts using actual
session boundaries as the observed reference. A connection is not a power draw.

For charging power, compare a prediction-based reconstruction with a
record-based reconstruction using the same assumed charger ratings and explicit
energy/window constraints. Report infeasible energy-window combinations and
unserved energy; do not silently clip them. Delivered grid kWh must not have
charging efficiency applied a second time. Dataset 4 modeled hourly data is
not independent measured load ground truth.

Do not combine plug-out-time next-event forecasts with actual future plug-in
features and call the result a day-ahead forecast. State what is known at each
prediction step. Any fully prospective forecast requires future-session energy
and duration inputs available at that same forecast origin, or explicit sampled
assumptions. Label an event-conditioned reconstruction accordingly.

**Deliverable:** one plot showing how behavioural uncertainty affects connected
occupancy or assumed demand, with MAE and peak error as appropriate. This is
optional and must not delay the main ML evidence and presentation.

## Stage 6 — Presentation and completion criteria

Tell the story: question -> data and provenance -> prediction targets and
information available -> baselines and split protocol -> real-data results ->
history/uncertainty findings -> controlled synthetic study if used -> demand
implication if completed -> limitations and next steps.

Minimum complete output:

- At least one behaviour target evaluated reproducibly against strong baselines
  on real held-out data, with supported conclusions about practical limitations.
- Clear separation of experienced/new users, causal inputs and censoring limits.
- Machine-readable metrics/configuration and a small set of presentation figures.
- A labelled synthetic study only if it answers a missing research question.
- A finished presentation with no unsupported prediction or grid-impact claims.

Use the existing modules and numbered notebooks. Add reusable code only where
needed. Keep verification focused on leakage, labels and experiment accounting;
avoid extensive low-value unit tests, new dashboards or unrelated restructuring.

**Progress:** the [audit and corrected baseline results](ML_AUDIT_RESULTS.md)
are available for all three targets. The prior timing command remains
unreproduced. Use the audited command for current results.

**Diagnostics complete:** paired user resampling, history-length analysis and
next-gap error breakdown are documented in the audit, with two presentation
figures. Connection duration has the clearest supported gain. A controlled
synthetic study remains optional for isolating history length from schedule
regularity.
