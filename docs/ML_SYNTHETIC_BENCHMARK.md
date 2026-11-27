# Calibrated synthetic connection-duration benchmark

The benchmark tests whether method rankings change when drivers share
charging routines, follow their own routines, or vary their behaviour from
one session to the next. Connection duration is elapsed time from plug-in
to unplugging.

```powershell
.venv/Scripts/python.exe -m src.models.synthetic_benchmark
```

The default study uses 1,000 simulated drivers per dataset, 48 sessions per
driver, three scenarios, and seeds 42–44. The same task, split proportion,
methods and checkpoints are used for each dataset. `--users`, `--sessions`
and `--seeds` control the size and replications. Synthetic IDs are local to
each file. The time axis is simulated local clock time.

## Real-data calibration

The generator reads only the training users and periods from the existing
[Norway pilot](ML_SIMILAR_USERS.md) and
[Electric Nation benchmark](methodology/electric_nation_intake.md).
It keeps calibration sessions with 0.25–72 h durations and 0.5–100 kWh
reported energy, and requires 15 qualifying sessions per reference driver.
It retains paired clock time, duration and energy, plus the calendar-day
gap since the preceding eligible plug-in. Day gaps used by the simulation
are capped at ten days.

Each dataset draws half its simulated users from each source. Within a source,
four clusters group reference drivers by timing, duration, overnight and
weekend tendencies, and median cadence. A simulated driver draws from their
assigned cluster, an individual reference profile, or the whole source pool.

| Scenario | Cluster | Individual | Whole source |
|---|---:|---:|---:|
| Shared routines | 85% | 10% | 5% |
| Individual routines | 10% | 85% | 5% |
| Irregular routines | 10% | 10% | 80% |

The generator adds random clock and duration variation. Irregular routines
get more clock variation. A sampled start shifts by whole days when needed
to avoid overlap, which can alter its weekday/weekend category. These are
controlled scenarios with empirically informed ranges and fixed session
counts. Delivered energy is paired with the sampled session template and
is contextual here. The experiment does not evaluate energy prediction
or infer battery SOC.

## Evaluation

The training and test cohorts have different simulated users and nonoverlapping
time periods. At 1, 3, 5 and 10 completed sessions, every method predicts the
same next connection duration. Models see the source category and their
usual calendar/history features. Cluster labels and source driver IDs stay
inside the generator. Donor histories use completed training sessions only,
and training-time neighbour features exclude the driver being predicted.
The two enriched boosting variants use identical training checkpoints.

Within each scenario, rank by mean absolute error (MAE) across seeds.
Per-run metrics include paired user-bootstrap intervals versus original
boosting and, for the hybrid variants, versus personal-history boosting.
The across-seed table reports mean MAE, range, mean rank and number of wins.
These results describe the simulated assumptions. The observed Norway and
UK results remain separate evidence.

Generated datasets are under `data/synthetic/calibrated_charging/`.
Result tables and calibration metadata are under
`outputs/tables/synthetic_benchmark/`.

Each compressed CSV contains an ordinal synthetic session ID and driver ID,
the calibration source under `location`, the train/test cohort, simulated
plug-in and plug-out timestamps (`origin` and `available`), elapsed duration
in hours (`target`), and contextual session energy (`energy_kwh`). The source
label identifies the calibration population, not a geographic charging site.

## Results

The nine files contain 432,000 generated sessions and 9,000 simulated driver
profiles across the nine runs. Each run produces 800 matched forecasts from
200 held-out drivers across four checkpoints, for 7,200 forecasts in total.
The calibration pools use 152 Norway users and 226 Electric Nation participants.

Lowest mean MAE across three seeds:

| Scenario | 1 completed session | 3 completed sessions | 5 completed sessions | 10 completed sessions |
|---|---|---|---|---|
| Shared | Boosting, 5.99 h | Boosting, 6.80 h | Boosting, 6.03 h | Personal history, 5.46 h |
| Individual | Personal history, 6.13 h | Boosting, 6.02 h | Boosting, 6.42 h | Boosting, 5.81 h |
| Irregular | Calendar median, 6.36 h | Calendar median, 6.24 h | Boosting, 6.15 h | Boosting, 6.69 h |

Original boosting leads in eight of the twelve average rankings and wins
15 of the 36 individual seed/checkpoint comparisons. At one prior session,
fixed blending reduces error versus the driver's own median by 2.35–3.09 h
across the three scenarios, but remains behind original boosting. Adding
similar-user features to richer personal-history boosting produces no
consistent gain. The calendar median has a small average advantage over
original boosting in the irregular scenario at one and three prior sessions
(0.02–0.04 h). Rank differences of that size should be read alongside
the per-seed scores and paired intervals.

See `outputs/figures/audited_ml/05_synthetic_benchmark.png` for the error curves.
