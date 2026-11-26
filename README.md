# BTP — EV User Behaviour and Grid Impact Modelling

B.Tech Project in Electrical Engineering, IIT Delhi. The project studies how travel and charging behaviour can be represented as EV demand and, where data and infrastructure permit, assessed for power-system impacts.

## Project progress

**Active research priority:** meaningful EV user-behaviour prediction. Follow
[`docs/ML_COMPLETION_PLAN.md`](docs/ML_COMPLETION_PLAN.md): reproduce and audit
existing ML, evaluate timing/energy against strong baselines, use synthetic data
for controlled studies where needed, and prepare the presentation. Charging
policy comparisons are supporting results; grid integration is deferred.

This tracker follows the deliverables in [`README_BTP_Five_Day_Plan.md`](README_BTP_Five_Day_Plan.md). Status is based on files and outputs present in this repository; it does not imply that a result is final or publication-ready. Update this table and the evidence below whenever a feature or deliverable is added or materially revised.

| Plan stage | Status | Remaining work |
|---|---|---|
| Day 1 — Audit and reproduce | Complete | [`docs/PROJECT_AUDIT.md`](docs/PROJECT_AUDIT.md) records repository/data findings, pipeline status, run guidance, reproducibility issue, and a recommended next ML target. |
| Day 2 — Defensible ML experiments | Corrected baseline benchmark complete | The [ML audit](docs/ML_AUDIT_RESULTS.md) reports current results for all three targets. Uncertainty and history-length diagnostics remain. |
| Day 3 — Charging-demand comparison | Implemented; see updated presentation priority | Immediate versus 23:00 charging and 3.7/7.4/11 kW sensitivity use one common feasible Home-to-Home cohort. Accounting, exclusions and scenario limits are documented in `docs/methodology/charging_policy_comparison.md`. ML integration remains separate. |
| Day 4 — Grid integration or bounded alternative | Not started | Audit for an existing feeder, base loads, bus mapping, and power-flow case. If absent, document the required data and a bounded demand-level result; do not claim grid impacts. |
| Day 5 — Presentation and rehearsal | Not started | Build and rehearse the 12–14 slide presentation with sourced, reproducible figures and clearly labelled measured, derived, inferred, and assumed quantities. |

**Progress summary:** Data exploration, charging-data analysis, the EV prototype and a fixed-cohort charging-policy comparison are present. The audited ML benchmark now provides corrected timing and energy results. Further ML diagnostics and the final presentation remain.

### Completed work and evidence

- **Charging policy comparison:** `python -m src.charging.compare_policies` writes policy metrics, cohort exclusions and profiles to `outputs/tables/charging_comparison/`, with presentation figures under `outputs/figures/charging_comparison/`. See [`docs/methodology/charging_policy_comparison.md`](docs/methodology/charging_policy_comparison.md) for the identical parking assumptions, fixed cohort and charger sensitivity.
- **Charging baseline validation (2026-11-26):** `python -m src.charging.validate_prototype` reproduces both existing aggregates for 6,136 feasible vehicle-days and saves accounting checks under `outputs/tables/prototype_validation/`. Four synthetic terminal-charging tests pass. See [`docs/methodology/prototype_validation.md`](docs/methodology/prototype_validation.md) for counts, limits and the next comparison; ML reproduction remains pending.
- **NHTS inspection and mobility EDA:** numbered notebooks `01_data_inspection.ipynb` and `02_travel_eda.ipynb` inspect the source tables and produce travel-pattern summaries and figures.
- **Trip chains and parking intervals:** `03_trip_chains.ipynb` builds candidate vehicle-day chains and interim internal gaps. Variable mappings and caveats are documented in [`docs/methodology/`](docs/methodology/).
- **Norway charging data analysis:** `04_ev_charging_data_analysis.ipynb` analyses reported charging sessions, timing, duration, energy, and user variation separately from NHTS. It does not fuse those sessions with NHTS vehicle days.
- **EV energy and charging prototype:** `src/charging/simple_ev.py` and `05_ev_load_prototype.ipynb` implement transparent trip energy, SoC accounting, and Home charging under configurable assumptions.
- **Two aggregate profiles:** the observed-window scenario uses only reconstructed internal NHTS Home gaps. The separate synthetic overnight scenario assumes terminal Home parking until the same vehicle-day's first-departure clock time the next day. The assumption and midnight handling are documented in [`docs/methodology/simple_ev_prototype.md`](docs/methodology/simple_ev_prototype.md). Figures are saved under `outputs/figures/ev_load_prototype/`.
- **Repository audit:** [`docs/PROJECT_AUDIT.md`](docs/PROJECT_AUDIT.md) summarizes the current pipeline, data sources/counts, execution notes, and the rationale for the session-energy experiment.
- **Session-energy ML experiment:** [`notebooks/06_ml_session_energy.ipynb`](notebooks/06_ml_session_energy.ipynb) and [`src/models/session_energy.py`](src/models/session_energy.py) predict reported energy for a future Norway residential charging session. [`docs/ML_EXPERIMENT.md`](docs/ML_EXPERIMENT.md) reports chronological and unseen-user results separately, with metrics in `outputs/ml_metrics.csv` and a held-out plot in `outputs/ml_actual_vs_predicted.png`.
- **Plug-in and plug-out time ML experiment:** [`notebooks/07_ml_plugin_plugout_times.ipynb`](notebooks/07_ml_plugin_plugout_times.ipynb) and [`src/models/charging_times.py`](src/models/charging_times.py) predict plug-out duration (`connection_time`) via two-stage overnight classification and next plug-in timing (`next_gap_hours`) using causal user history, calendar context, and empirical 2022 NHTS mobility priors. Provides $p_{10}, p_{50}, p_{90}$ quantile uncertainty envelopes, evaluated on chronological (warm and cold-start users) and unseen-user holdouts. Metrics are in `outputs/ml_charging_times_metrics.csv` and diagnostic plots in `outputs/`.
- **Central ML documentation:** [`docs/ML_README.md`](docs/ML_README.md) synthesises both ML models, causal feature engineering, leakage controls, and comparative findings.

### Current scope limits

- The EV parameters and overnight terminal stay are assumptions, not NHTS observations.
- Aggregate vehicle-day curves are unweighted sample simulations, not a regional or national demand forecast.
- Norway charging data and NHTS mobility data are bridged via empirical prior distributions; individual records remain separate.
- The ML results predict recorded residential session energy, connection duration, and inter-session gap hours. They do not predict nationwide EV adoption or full power-system bus voltages yet.
- The repository has no final presentation yet.

## Intended pipeline

`mobility data → trip chains → EV energy and SoC → charging behaviour → individual and aggregate demand → grid integration`

Keep observed data, derived fields, inferred quantities, and synthetic assumptions distinguishable. Keep each stage replaceable as the research methodology develops.

## Repository layout

- `data/` — raw source data and interim/processed analysis files; preserve raw originals.
- `notebooks/` — numbered inspection, EDA, trip-chain, charging-data, and EV prototype work.
- `src/` — reusable data, behaviour, charging, grid, model, and visualization packages as they develop.
- `docs/` — methodology, variable mappings, literature, and project notes.
- `outputs/` — generated figures, tables, and presentation artifacts.
- `README_BTP_Five_Day_Plan.md` — five-day delivery plan and operating constraints.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open a notebook from the repository root, for example:

```powershell
jupyter notebook notebooks/05_ev_load_prototype.ipynb
```

## Updating this tracker

After adding a feature, update its stage status and remaining work above, add or revise the evidence bullet, and adjust the progress summary. Mark a plan deliverable complete only when its stated outputs are present and reproducible; keep partial work labelled partial. Do not count future roadmap ideas as completed work.
