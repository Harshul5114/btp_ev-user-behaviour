# Project audit and next-step recommendation

**Audit date:** 2026-11-23  
**Scope:** repository state, data provenance and schemas, existing analysis pipeline, reproducibility status, and one ML target recommendation. No ML code is implemented by this audit.

## Executive summary

The repository contains a reproducible-looking NHTS mobility workflow, a separate analysis of residential EV charging sessions from Norway, and an NHTS-based EV charging prototype with observed-window and synthetic overnight aggregates. The prototype is a sample-level rule-based simulation; it does not predict behaviour, fuse the Norway and NHTS datasets, or calculate power-system impacts.

The completed ML experiment predicts delivered energy for a future residential charging session from Norway Dataset 1. Dataset 1 has 35,377 reported sessions across 267 repeat-user IDs at 12 locations. All 35,377 have parsed plug-in times and recorded session energy. The experiment uses a chronological 80/20 session-time holdout and a separate unseen-user holdout. Its design, results and limitations are documented in `docs/ML_EXPERIMENT.md`.

## Repository map

| Area | Current contents and role |
|---|---|
| `data/raw/nhts_2022/` | NHTS 2022 Household, Person, Vehicle, Trip and Long Distance public CSVs, plus official codebook, user guide and citation PDF. Raw files are Git-ignored. |
| `data/raw/norway_ev/` | Four supplied charging-report/prediction CSVs and the dataset paper. Raw files are Git-ignored. |
| `data/raw/Level 3 EV charging dataset/` | Additional measurement and session workbooks. No analysis pipeline using these workbooks was identified in the numbered notebooks reviewed. |
| `data/interim/` | `vehicle_trip_chains.csv` and `vehicle_parking_intervals.csv`, produced by notebook 03. They are Git-ignored. |
| `data/processed/` | No processed analysis dataset currently present (only `.gitkeep`). |
| `notebooks/01_data_inspection.ipynb` | Structural inspection of the NHTS tables, keys, codes, missingness and candidate driver scope. |
| `notebooks/02_travel_eda.ipynb` | NHTS travel timing, distance, destination, day type, vehicle-day and dwell EDA. |
| `notebooks/03_trip_chains.ipynb` | Candidate vehicle trip-chain and internal parking-gap derivation; writes interim CSVs and trip-chain figures. |
| `notebooks/04_ev_charging_data_analysis.ipynb` | Standalone audit and EDA of Norway charging reports and their associated model-derived tables. |
| `notebooks/05_ev_load_prototype.ipynb`, `src/charging/simple_ev.py` | Configurable rule-based EV energy/SoC and home-charging simulation; compares observed internal Home gaps with a separately assumed synthetic terminal Home stay. |
| `docs/methodology/` | NHTS mappings, trip-chain schema, Norway provenance/use note, and EV prototype assumptions. |
| `outputs/figures/` | Existing travel EDA, trip-chain, Norway charging, and EV prototype plots. The two EV aggregate figures are `02_aggregate_home_charging_sample.png` and `03_aggregate_home_charging_synthetic_overnight.png`. |
| `outputs/tables/`, `outputs/presentation/` | No generated tables or presentation identified; only `.gitkeep` files. |
| `src/models/`, `src/grid/`, other `src/` packages | Package placeholders; no ML or grid model implementation identified. |
| `tests/` | Only `.gitkeep`; no executable tests identified. |

## Data and schema findings

### NHTS 2022

The official files currently in `data/raw/nhts_2022/` include `hhv2pub.csv` (7,893 rows, 43 columns), `perv2pub.csv` (16,997 rows, 153 columns), `vehv2pub.csv` (14,684 rows, 55 columns), and `tripv2pub.csv` (31,074 rows, 102 columns). The separate `ldtv2pub.csv` is not part of the day-travel analysis. Notebook 01 reports unique keys and successful household/person/vehicle joins for the inspected records. Exact source-field meanings and encoded values are mapped in `docs/methodology/nhts_variable_mapping.md`, using the bundled official codebook and guide.

Notebook 03 derives 19,673 candidate driver legs across 6,874 identified household vehicles after holding aside invalid measurements and conflicting same-vehicle reports. It exports 12,799 between-leg gaps. The schema note reports 12,159 analysis-eligible internal gaps after the chain and survey-window filters. These are derived candidate trajectories from person-reported trip records, not GPS-verified vehicle motion. `TDAYDATE` is a month (`YYYYMM`), so it cannot supply a true next-day date. NHTS trip weights are not automatically valid weights for constructed vehicle-days or parking intervals.

### Norway residential charging data

The provenance note `docs/methodology/norway_ev_dataset_note.md` identifies 12 residential/shared-residential locations, with reporting periods spanning 2018–2021. It distinguishes Dataset 1 reported session values from Dataset 2 user predictions, Dataset 3 session-level derived/predicted values, and Dataset 4 modeled hourly values. The last three must not be mistaken for independent telemetry or observed battery SoC.

| Table | Rows | Grain | Audit notes |
|---|---:|---|---|
| Dataset 1 charging reports | 35,377 | One reported session; `session_id` unique | 267 users, 12 locations. `energy_session` is reported delivered energy; `plugin_time` is present for all rows. 38 lack plug-out time/connection duration; 439 have energy at or below 0.5 kWh despite the paper's reported cleaning criterion, so they are retained and flagged by the existing analysis. |
| Dataset 2 user predictions | 267 | One row per user | Six users lack both predicted charging power and inferred battery capacity. Do not use these as observed hardware truth. |
| Dataset 3 session predictions | 34,537 | One row per predicted session | 261 users; contains predicted/assumption-based SoC fields and charging/idle-time calculations. |
| Dataset 4 hourly predictions | 469,897 | Modeled session-hour rows | 34,537 identified sessions plus 6,710 rows with no session/user ID; hourly energy and SoC fields are model outputs. |

The raw Norway CSVs are semicolon-delimited with comma decimal separators. The existing notebook specifies both when loading. Dataset 1 spans 2018-02-06 through 2021-08-05 by parsed plug-in time. See the dataset note for daylight-saving caveats, duration discrepancies and limits on representativeness.

## Pipeline and outputs status

| Stage | Observed repository state |
|---|---|
| Source inspection and mappings | Implemented in notebooks 01 and 04; supporting mappings/provenance notes are present. |
| NHTS travel EDA | Implemented in notebook 02 with saved figures. |
| Candidate trip chains and internal gaps | Implemented in notebook 03 with interim outputs and saved figures. Chain continuity is based on reported purpose continuity and has documented limits. |
| Norway charging EDA | Implemented separately in notebook 04 with saved figures. No NHTS–Norway fusion. |
| Rule-based EV energy and charging | Implemented in `src/charging/simple_ev.py` and notebook 05. Current assumptions are 60 kWh, 80% initial SoC, 0.18 kWh/km, 7.4 kW grid-side Home charging, 90% efficiency and 90% target SoC. |
| Aggregate comparison | Implemented for the same 6,136 feasible sample vehicle-days. Embedded latest notebook output reports 10,767.9 kWh and 1,240.8 kW at 12:45–13:00 for observed-window-only charging; synthetic overnight reports 86,606.0 kWh and 10,138.1 kW at 18:00–18:15. The synthetic scenario adds 75,838.1 kWh under its terminal-parking assumption. These are unweighted sample simulations, not population forecasts. |
| ML behaviour prediction | Implemented only for reported Norway residential session energy. `src/models/session_energy.py` and notebook 06 provide chronological and unseen-user evaluations; see `docs/ML_EXPERIMENT.md`. |
| Grid mapping/power flow | Not implemented. `src/grid/` is a placeholder; no feeder, bus mapping, base-load profile or no-EV power-flow case was identified. |
| Final presentation | Not identified under `outputs/presentation/`. |

## Reproduction guidance and execution status

Create the project environment and install the declared dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run notebooks from the repository root in numerical order where later notebooks need earlier interim outputs:

```powershell
jupyter notebook notebooks/01_data_inspection.ipynb
jupyter notebook notebooks/02_travel_eda.ipynb
jupyter notebook notebooks/03_trip_chains.ipynb
jupyter notebook notebooks/04_ev_charging_data_analysis.ipynb
jupyter notebook notebooks/05_ev_load_prototype.ipynb
```

Notebook 03 must be run before notebook 05 if the ignored interim CSVs are absent or need regeneration. Notebooks create/overwrite figures and notebook outputs; preserve reviewed artifacts before a full rerun.

The local `.venv` has Python 3.13.3 and pinned dependencies in `requirements.txt`, including scikit-learn 1.9.1 for the completed ML experiment. There is no test suite to run.

The EV prototype notebook was executed top-to-bottom in the current working session before this audit, and its embedded outputs and both aggregate figures are present. On this Windows environment the notebook kernel initially failed while setting Jupyter runtime-file permissions (`WinError 5`); it completed when run with `JUPYTER_ALLOW_INSECURE_WRITES=1` and a workspace-local runtime directory. This is an environment-specific workaround; do not enable insecure writes globally. Other notebooks have saved outputs/figures in the repository, but this audit did not rerun all of them because they write over existing output artifacts.

## Selected ML target — future-session delivered energy

**Recommendation:** predict `energy_session` (kWh) for a future residential charging report, using information available when a session starts. This is a direct continuous target in Dataset 1 and maps to an interpretable uncertainty in charging demand. Treat the work as a prediction of a recorded session at these sampled sites, not a prediction of whether a vehicle will charge or a national EV-demand forecast.

### Target, cohort and usable counts

- **Target:** Dataset 1 `energy_session`, the reported energy delivered during one charging report, in kWh.
- **Prediction point:** session plug-in. Candidate inputs must be known at or before `plugin_time`.
- **Usable rows from the current raw file:** 35,377 of 35,377 have nonmissing `energy_session` and parseable `plugin_time`; there are 267 distinct `user_id` values and 12 locations. The source analysis reports positive energies throughout. These counts are before any task-specific exclusions or feature-generation rules.
- **User history:** the existing session-count summary has a median of 77 reports/user (range 10–872). A past-only user history feature is therefore feasible for many users, with a cold-start fallback required for users with no prior training session.

### Candidate features available at plug-in

- Local plug-in hour, day of week, month/season derived from `plugin_time`.
- `location` (12 sampled sites), handled categorically.
- User's past-only session-energy history, such as count and median/mean of earlier sessions; compute each feature strictly from records before the prediction timestamp. Use a training/site/time fallback for a user with no prior history.

Do **not** include `energy_session`-derived features, `plugout_time`, `connection_time`, or Dataset 2–4 charging-power, battery, charging-time, idle-time, hourly-energy or SoC predictions. Those are target leakage, unavailable at plug-in, or model-derived quantities. Avoid global target encoding or summary statistics computed using future/test sessions.

### Evaluation split

Use a chronological split of Dataset 1 sorted by parsed `plugin_time`: earliest 80% for development/training and latest 20% as the future-session test set. On the current file, the timestamp cutoff is approximately **2020-01-26 00:14:51**: 28,301 training sessions from 220 users and 7,076 test sessions from 228 users. Of the test set, 3,643 sessions from 181 users have user history represented in training; 3,433 sessions from 47 test-only users are cold-start cases. Report warm-user and cold-user performance separately, and do not mix them into one unexplained metric. Any tuning/validation split should also be chronological within the training period.

This split tests later timestamps in the supplied dataset, not transport to new sites, regions or years. Site collection periods may differ, so report site coverage across the cutoff and inspect whether test performance is dominated by site or time shifts. Use MAE and RMSE in kWh, compare against a training-median baseline, and keep all preprocessing and history features causal.

### Main leakage and interpretation risks

1. Repeated sessions from the same user are correlated. A random row split would let later sessions inform prediction of earlier sessions and overstate generalization.
2. User history is valid only when built from strictly prior sessions; the held-out target or future records must never enter its summary.
3. Plug-out time and connection duration occur after plug-in and can reveal charging opportunity; they are invalid for this prediction point.
4. Dataset 2–4 fields are predictions/derived simulations and may encode assumptions or the target itself. They are not independent measured features.
5. Users and sites come from a small, nonrepresentative residential sample. Performance does not imply behaviour for Norwegian EV owners generally, workplaces/public sites, or NHTS households.
6. Session energy predicts energy conditional on a recorded session. It does not predict session occurrence, charging location choice, or daily EV energy demand.

## Next action

Use the completed experiment only as a documented residential-session result. The next plan stage is the charging-demand comparison: assess whether its observed session-energy distribution can inform a defensible scenario without treating it as NHTS observations or silently fusing the datasets.
