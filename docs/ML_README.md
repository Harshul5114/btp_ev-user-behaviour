# Machine Learning in EV User Behaviour & Charging Modelling

The [audited real-data results](ML_AUDIT_RESULTS.md) are the current benchmark.
The experiments below record the previous protocol and its reported scores.
See the audit for the differences in label availability, splits and baselines.

This document synthesises the machine-learning models developed for the IIT Delhi Electrical Engineering B.Tech Project (BTP).

The project studies how travel behaviour and residential EV charging data can be translated into spatial-temporal charging demand profiles for power-system analysis. Rather than replacing physical energy conservation or network power-flow physics with a black-box model, ML is deployed specifically to learn **uncertain user behaviour**:
1. **Delivered charging energy** per session ($E_i$, kWh).
2. **Session connection duration / plug-out timing** ($C_i$, hours $\rightarrow T_{\text{out}}$).
3. **Inter-session idle gap / next plug-in timing** ($NG_i$, hours $\rightarrow T_{\text{in}, i+1}$).

---

## 1. Overview of ML Experiments

| Experiment | Target Variable | Prediction Moment | Primary Model & Setup | Deliverables & Artifacts |
|---|---|---|---|---|
| **Session Energy** | `energy_session` (kWh) | Plug-in time ($T_{\text{in}}$) | `HistGradientBoostingRegressor` (squared error loss), causal user energy history | [`notebooks/06_ml_session_energy.ipynb`](../notebooks/06_ml_session_energy.ipynb)<br>[`src/models/session_energy.py`](../src/models/session_energy.py)<br>[`outputs/ml_metrics.csv`](../outputs/ml_metrics.csv)<br>[`outputs/ml_actual_vs_predicted.png`](../outputs/ml_actual_vs_predicted.png) |
| **Plug-out Time** | `connection_time` (hours) | Plug-in time ($T_{\text{in}}$) | Two-stage overnight-aware classifier $P(\text{overnight})$ + `HistGradientBoostingRegressor` ($L_1$ point loss & $p_{10}, p_{50}, p_{90}$ quantile losses) | [`notebooks/07_ml_plugin_plugout_times.ipynb`](../notebooks/07_ml_plugin_plugout_times.ipynb)<br>[`src/models/charging_times.py`](../src/models/charging_times.py)<br>[`outputs/ml_charging_times_metrics.csv`](../outputs/ml_charging_times_metrics.csv)<br>[`outputs/ml_connection_time_actual_vs_predicted.png`](../outputs/ml_connection_time_actual_vs_predicted.png)<br>[`outputs/ml_charging_times_by_dayofweek.png`](../outputs/ml_charging_times_by_dayofweek.png) |
| **Next Plug-in Time** | `next_gap_hours` (hours) | Plug-out time ($T_{\text{out}}$) | `HistGradientBoostingRegressor` ($L_1$ point loss & $p_{10}, p_{50}, p_{90}$ quantile losses), circular clock-hour recovery | [`notebooks/07_ml_plugin_plugout_times.ipynb`](../notebooks/07_ml_plugin_plugout_times.ipynb)<br>[`src/models/charging_times.py`](../src/models/charging_times.py)<br>[`outputs/ml_charging_times_metrics.csv`](../outputs/ml_charging_times_metrics.csv)<br>[`outputs/ml_next_gap_actual_vs_predicted.png`](../outputs/ml_next_gap_actual_vs_predicted.png) |

---

## 2. Leakage Controls & Causal Formulation

All models enforce strict causal isolation:
* **No Future Leakage:** A session $k$ never uses its own or any subsequent session's realised duration, energy, or departure time.
* **Chronological Splits:** Evaluations use a strict 80/20 chronological partition (cutoff date: **2020-01-26 01:22:41** for complete records; **2020-01-26 00:14:51** for energy records).
  - *Warm users:* Users observed in training who continue charging in the test period.
  - *Cold-start users:* New users who first appear after the chronological cutoff.
* **Unseen-User Generalization:** A separate 20% user holdout via `GroupShuffleSplit(random_state=42)` evaluates zero-history cold-start performance across entire unseen residential populations.
* **No Target Proxies:** Post-plug-in fields (`plugout_time`, `connection_time`, `energy_session`) are excluded from plug-in-time features. Model-derived quantities in Datasets 2–4 (e.g. synthetic battery capacities, simulated hourly power) are strictly excluded.
* **NHTS Empirical Mobility Priors:** A 48-bin lookup table indexed by $(\text{hour}, \text{is\_weekend})$ from 2022 NHTS vehicle chains anchors charging models with national travel distributions:
  - $P(\text{arrive home at hour } h \mid \text{daytype})$
  - $P(\text{first depart home at hour } h \mid \text{daytype})$
  - $\mathbb{E}[\text{Home stay duration} \mid \text{hour}, \text{daytype}]$
* **Site Context:** Smoothed out-of-fold target encoding for `location` fit on training splits with empirical Bayes $m=10$ smoothing.

---

## 3. Consolidated Results Summary

### Experiment 1: Session Delivered Energy (`energy_session`, kWh)

Target: reported kWh per charging event (Norway Dataset 1, 35,377 sessions).

| Evaluation | Subset | Predictor | Sessions | Users | MAE (kWh) | RMSE (kWh) |
|---|---|---|---:|---:|---:|---:|
| Chronological Future | All Test | Training median | 7,076 | 228 | 8.79 | 13.31 |
| Chronological Future | All Test | Causal user-history median | 7,076 | 228 | 5.65 | 8.90 |
| Chronological Future | All Test | **Gradient Boosting** | 7,076 | 228 | **5.55** | **8.23** |
| Chronological Future | Warm Users | **Gradient Boosting** | 3,643 | 181 | **4.83** | **7.42** |
| Chronological Future | Cold-Start | **Gradient Boosting** | 3,433 | 47 | **6.32** | **9.00** |
| Unseen-User Holdout | All Test | Training median | 8,894 | 54 | **6.64** | **9.78** |
| Unseen-User Holdout | All Test | Gradient Boosting | 8,894 | 54 | 8.15 | 10.06 |

*Key finding:* User history is a dominant predictor of delivered energy. For warm users, gradient boosting improves over the history baseline. For unseen users, the training median baseline outperforms the model, emphasizing that energy habits cannot be reliably inferred for new drivers without prior telemetry.

---

### Experiment 2: Plug-Out Time & Connection Duration (`connection_time`, hours)

Target: duration between plug-in and unhook (35,339 complete sessions).

| Evaluation | Subset | Predictor | MAE (h) | RMSE (h) | Circular Hour MAE (h) | 80% Coverage | Mean Width (h) |
|---|---|---|---:|---:|---:|---:|---:|
| Chronological Future | All Test | Training median | 9.74 | 16.34 | 6.10 | 75.9% | 20.01 |
| Chronological Future | All Test | Causal user-history median | 8.38 | 14.62 | 4.65 | 75.9% | 20.01 |
| Chronological Future | All Test | **Hist Gradient Boosting** | **7.67** | **14.65** | **3.82** | **77.4%** | 21.91 |
| Chronological Future | Warm Users | Training median | 10.23 | 17.21 | 5.94 | 74.7% | 20.01 |
| Chronological Future | Warm Users | Causal user-history median | 8.73 | 14.77 | 4.71 | 74.7% | 20.01 |
| Chronological Future | Warm Users | **Hist Gradient Boosting** | **8.20** | **15.27** | **3.86** | **79.2%** | 23.92 |
| Chronological Future | Cold-Start | **Hist Gradient Boosting** | **7.12** | **13.97** | **3.78** | **75.5%** | 19.79 |
| Unseen-User Holdout | All Test | Training median | 8.36 | 13.71 | 5.97 | 77.7% | 20.66 |
| Unseen-User Holdout | All Test | **Hist Gradient Boosting** | **7.72** | **14.53** | **4.32** | **75.1%** | 16.48 |

*Key finding:* Two-stage overnight-aware regression and calendar features reduce connection duration MAE by **21.2%** over the training median. Recovered plug-out clock hour circular error is **3.82 hours**, and quantile coverage on warm users is **79.2%** (closely matching the 80% nominal interval). Unlike energy, connection duration shows strong predictability even for cold-start and unseen users due to shared commuter schedule constraints.

---

### Experiment 3: Next Plug-In Timing (`next_gap_hours`, hours)

Target: duration from current plug-out to next plug-in event.

| Evaluation | Subset | Predictor | MAE (h) | RMSE (h) | Circular Hour MAE (h) | Pinball $p_{90}$ | 80% Coverage |
|---|---|---|---:|---:|---:|---:|---:|
| Chronological Future | All Test | Training median | 53.34 | 158.96 | 5.50 | 29.51 | 73.1% |
| Chronological Future | All Test | Causal user-history median | 47.65 | 157.19 | 4.82 | 29.51 | 73.1% |
| Chronological Future | All Test | **Hist Gradient Boosting** | **42.08** | **143.05** | **5.00** | **18.89** | **75.2%** |
| Chronological Future | Warm Users | Training median | 50.79 | 172.73 | 5.31 | 28.64 | 77.8% |
| Chronological Future | Warm Users | Causal user-history median | 45.83 | 175.80 | 4.59 | 28.64 | 77.8% |
| Chronological Future | Warm Users | **Hist Gradient Boosting** | **39.60** | **156.28** | **4.88** | **18.24** | **78.5%** |
| Chronological Future | Cold-Start | **Hist Gradient Boosting** | **44.62** | **128.13** | **5.11** | **19.54** | **71.8%** |
| Unseen-User Holdout | All Test | Training median | 28.08 | 93.02 | 6.04 | 15.69 | 81.3% |
| Unseen-User Holdout | All Test | **Hist Gradient Boosting** | 29.70 | **89.82** | 6.49 | 17.07 | 75.9% |

*Key finding:* Predicting multi-day idle intervals between sessions is inherently heavy-tailed. Gradient boosting reduces warm-user MAE from 50.79 h to **39.60 h**, and cuts the 90th percentile pinball loss from 28.64 to **18.24**, sharply reducing over-prediction of long inter-session gaps.

---

## 4. Reproducing the Experiments

All models run within the project virtual environment:

```powershell
# 1. Run Session Energy ML Experiment
python -m src.models.session_energy

# 2. Run Plug-in & Plug-out Times ML Experiment
python -m src.models.charging_times
```

Interactive analyses with embedded plots and metric tables are in:
- [`notebooks/06_ml_session_energy.ipynb`](../notebooks/06_ml_session_energy.ipynb)
- [`notebooks/07_ml_plugin_plugout_times.ipynb`](../notebooks/07_ml_plugin_plugout_times.ipynb)

---

## 5. Interface with Downstream Grid Load Simulation

In power-flow simulation (Days 3–4), these ML outputs inform individual charging schedules:
1. When a vehicle unplugs at time $T_{\text{out}}$, `next_gap_hours` determines when the next charging load will arrive at the residential bus.
2. When the vehicle plugs in at $T_{\text{in}}$, `connection_time` bounds the physical availability window $[T_{\text{in}}, T_{\text{out}}]$, while `energy_session` determines total electrical energy demanded.
3. The power draw $P_i(t)$ is constrained by charger rating $P_{\text{charger}}$ and charging duration $T_{\text{charge}} = \frac{E_i}{\eta \cdot P_{\text{charger}}}$. If $T_{\text{charge}} < C_i$, the vehicle idles connected, providing potential flexibility for smart charging / valley-filling dispatch.
