# BTP Five-Day Plan for Codex

## Objective

Deliver the strongest defensible final BTP result in five days.
Prioritize reproducibility, one evaluated ML task, charging-demand
comparison, and grid integration only if existing infrastructure
supports it.

`Mobility + charging data -> behaviour prediction -> individual P_i(t) -> aggregate P_EV(t) -> bus loads -> power-flow analysis`

ML should model uncertain user behaviour. Energy accounting, charger
constraints, aggregation and power flow remain explicit engineering
models.

## Last known state --- verify before relying on it

These details may be stale. Inspect the repository and current outputs
first.

-   **NHTS:** 2022 Household/Person/Vehicle/Trip tables; last reported
    19,673 candidate trips, 6,874 vehicles, 97.1% purpose continuity in
    multi-leg chains, 12,159 internal parking intervals. Findings
    included work-arrival concentration around 7--8 AM, home arrivals
    around 5--6 PM, median trip distance about 4.9 miles.
-   **Norway EV data:** last reported 35,377 residential charging
    sessions from 267 users. Analysed plug-in/out times, session energy,
    connection duration and user heterogeneity. Some battery/SoC fields
    may be inferred. Last known status: analysed separately, not fused
    with NHTS simulator.
-   **Current prototype assumptions:** 60 kWh battery, 80% initial SoC,
    0.18 kWh/km, 7.4 kW home charger, 90% efficiency, 90% target SoC.
    Rule: charge immediately at Home if SoC \< 90%, until target or
    parking interval ends.
-   **Last reported aggregate run:** about 6,225 vehicle-days, 6,136
    feasible, peak about 1.24 MW near 12:45 PM. This is a sample-level
    prototype output, not a population forecast.
-   **Overnight boundary:** final home arrival is observed, but next
    day's departure is absent from a one-day NHTS vehicle record. A
    synthetic scenario was proposed: next departure = today's first
    departure + 24 hours. Keep observed-internal and
    synthetic-terminal-parking scenarios separate.
-   **Literature:** six papers cover user behaviour,
    mobility/spatial-temporal charging, electrical EV load models and
    grid impact. Use supplied PDFs/notes for exact citations.

## Day 1 --- Audit and reproduce

1.  Inspect repo structure, README/AGENTS, Git status, notebooks,
    scripts, tests, configs and data schemas.
2.  Trace NHTS/Norway data through processing, plots, simulator and
    aggregate outputs.
3.  Run existing tests and smallest reproducible pipeline.
4.  Locate latest original and synthetic-overnight profiles.
5.  Record exact commands, dependencies, failures and authoritative
    results.

**Deliver:** `docs/PROJECT_AUDIT.md` with repo map, run commands,
data/schema summary, pipeline status, issues and recommended next step.

**Do not** reorganize the repo, rename/delete data, overwrite outputs or
refactor unrelated code during the audit.

## Day 2 --- One defensible ML experiment

Choose the target only after inspecting actual fields and sample
coverage. Candidate targets: session energy (kWh), connection duration,
or an unambiguous session-timing target.

1.  Define target, features, usable rows/users and prediction use case.
2.  Establish a simple baseline (mean/median, time-of-day or
    user-history baseline).
3.  Train one tabular model (Random Forest or Gradient Boosting/XGBoost
    if appropriate).
4.  Evaluate with MAE/RMSE (regression) or suitable classification
    metrics.
5.  Produce metrics table and actual-vs-predicted plot.
6.  Document errors and limitations.

**Leakage rules:** use chronological splits for future-session
prediction; use user-grouped holdout for unseen-user generalization;
report these as different tasks. Fit preprocessing on training data
only. Do not randomly split rows if it leaks future sessions or
repeated-user patterns. Do not use target proxies. Do not treat inferred
fields as ground truth.

**Deliver:** reproducible script/notebook, `outputs/ml_metrics.csv`,
`outputs/ml_actual_vs_predicted.png`, `docs/ML_EXPERIMENT.md`. Do not
claim the model predicts behaviour beyond its actual target.

## Day 3 --- Charging-demand comparison

1.  Preserve current immediate-home-charging model as baseline.
2.  Check SoC and energy accounting, units, charger-side efficiency and
    parking constraints.
3.  Introduce only defensible improvements: empirical
    heterogeneity/session-energy or duration distributions from Norway,
    or the ML prediction if it maps coherently into the simulator.
4.  Keep parking opportunity distinct from an actual charging event.
5.  Generate individual `P_i(t)` and aggregate `P_EV(t)`.
6.  Compare baseline vs improved scenario using same sample and time
    resolution.
7.  Report peak kW, peak time, total energy kWh and session/vehicle
    counts. Do not assume the improved curve must peak at night or be
    lower.

**Deliver:** scenario plot, metrics, updated simulator and
`docs/CHARGING_SIMULATION.md`.

## Day 4 --- Grid integration or bounded alternative

First inspect whether a working feeder/test network, no-EV power flow,
base loads, bus mapping and simulator already exist.

If available: 1. Reproduce no-EV baseline. 2. Map EV demand to buses
with an explicit documented assumption. 3. Add EV load at compatible
time resolution. 4. Compare voltage, branch loading and losses; report
only computed metrics.

If unavailable, do not build a complete grid framework from scratch in
one day. Produce a demand-level analysis and document the data/mapping
required for future grid integration.

**Deliver:** `docs/GRID_INTEGRATION.md` and reproducible comparison if
feasible.

## Day 5 --- Presentation and rehearsal

Aim for 12--14 slides, adapted to allotted time: 1. Motivation/problem
2. Literature synthesis/research gap 3. Overall framework 4. Datasets
and provenance 5. NHTS mobility findings 6. Trip chains/parking 7.
Norway charging behaviour 8. Methodology 9. ML target/model/evaluation
10. Charging simulation/scenario comparison 11. Grid result or
integration status 12. Limitations/assumptions 13. Conclusions 14.
Roadmap/questions

Use readable plots, units, captions and citations. Clearly distinguish
measured/survey-reported, derived, inferred and synthetic quantities.
Label completed/current/future work. Do not claim ML, NHTS--Norway
fusion or grid results that are not implemented. Do not present
sample-level simulated demand as a city/national forecast.

**Deliver:** editable PPTX, final plots/metrics, concise speaker
notes/Q&A and commands to reproduce key results.

## Codex operating instructions

Act as a careful research-engineering assistant. 1. Audit first; report
findings before coding. 2. Recommend the single highest-value ML target
supported by actual data, including target, features, sample/user
counts, split and leakage risks. 3. Explain planned file changes before
making them. 4. Make one logical change at a time; run relevant
tests/scripts. 5. Preserve raw data and existing outputs. 6. Do not
fabricate data, metrics, plots or citations. 7. Label synthetic
assumptions; do not silently replace real data with synthetic data. 8.
Do not commit secrets, credentials or personal data. 9. Avoid broad
refactors and unnecessary dependencies. 10. Ask before destructive,
expensive or irreversible operations. 11. If a task is not feasible in
five days, state that and propose a bounded alternative.

### First task for Codex

**Do not implement ML yet. Inspect the repository and create
`docs/PROJECT_AUDIT.md`. Then recommend one ML target that the actual
data supports, with its target definition, features, usable sample/user
counts, evaluation split and leakage risks. Wait for approval before
broad changes.**

## Definition of success

A successful five-day outcome means: reproducible existing analysis; one
correctly formulated and evaluated ML task; a charging-demand comparison
informed by that model where justified; grid analysis if existing
infrastructure permits; transparent assumptions/limitations; and a
presentation where every result can be explained and reproduced.

**Principle: build the smallest complete and defensible research story,
not the largest possible technical scope.**
