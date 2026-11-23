# Future-session EV charging-energy prediction

## Question and target

This experiment predicts the reported **delivered energy** for a future residential charging session in Norway. The target is Dataset 1 `energy_session`, in kWh. It predicts energy conditional on a session being recorded; it does not predict whether a driver will charge, where they will charge, battery SoC, or national EV demand.

Dataset 1 contains 35,377 usable reports: every row has a parsed `plugin_time` and a nonnegative recorded `energy_session`. The records cover 267 repeat-user IDs at 12 residential/shared-residential locations. These are not a representative sample of all Norwegian EV owners, and they should not be fused directly with NHTS vehicle-days.

## Prediction-time features

Every feature is available at plug-in time:

- local plug-in hour encoded as sine/cosine, weekday, weekend indicator, and calendar month encoded as sine/cosine;
- residential `location` as a categorical feature; and
- causal per-user past-session count, mean delivered energy, and median delivered energy.

Each history value uses only sessions with an earlier plug-in timestamp. Sessions sharing one timestamp are featurized before any of their target values update history. A user without prior history receives zero history values plus a zero count. The model sees no `user_id`, preventing it from memorizing individual labels.

The experiment excludes `plugout_time` and `connection_time` because they are only known after plug-in. It also excludes Dataset 2–4 predicted charging power, battery capacity, SoC, charging time, idle time, and hourly energy: they are derived/modelled quantities rather than independent measurements and may act as target proxies.

## Model and baselines

The tabular model is `HistGradientBoostingRegressor`, selected as a computationally efficient gradient-boosting model for mixed time, location, and causal-history features. Location is one-hot encoded from training data only, with unknown locations ignored safely at evaluation time.

Two baselines use no fitted ML model:

- **Training median:** median target energy from the training partition.
- **Causal user-history median:** median of the user's earlier observed session energies when available; otherwise the training median.

## Evaluation designs

### Future-session chronological evaluation

Records are ordered by plug-in time. The earliest 80% form training and the latest 20% form test, using a cutoff of **2020-01-26 00:14:51**. The split has 28,301 training sessions from 220 users and 7,076 later test sessions from 228 users.

The test set contains 3,643 sessions from 181 users represented in training (warm users), and 3,433 sessions from 47 users absent from training (cold-start users). For the future-session task, a later test session may use an earlier test session's realised energy in its user history, reflecting sequential deployment after prior sessions have finished. It never uses its own or a future session's energy.

### Unseen-user evaluation

`GroupShuffleSplit` with `random_state=42` holds out complete users: 54 users and 8,894 sessions form the test set; 213 users and 26,483 sessions form training. Test-user histories are unavailable by design, so their history baseline falls back to the training median. This answers a different question from the chronological result and must be interpreted separately.

## Results

MAE and RMSE are in kWh. The complete machine-readable table is [`outputs/ml_metrics.csv`](../outputs/ml_metrics.csv); [`outputs/ml_actual_vs_predicted.png`](../outputs/ml_actual_vs_predicted.png) visualises gradient-boosting predictions on the chronological test set.

| Evaluation | Test subset | Predictor | Sessions | Users | MAE (kWh) | RMSE (kWh) |
|---|---|---|---:|---:|---:|---:|
| Future chronological | All | Training median | 7,076 | 228 | 8.79 | 13.31 |
| Future chronological | All | Causal user-history median | 7,076 | 228 | 5.65 | 8.90 |
| Future chronological | All | Gradient boosting | 7,076 | 228 | **5.55** | **8.23** |
| Future chronological | Warm users | Gradient boosting | 3,643 | 181 | 4.83 | 7.42 |
| Future chronological | Cold-start users | Gradient boosting | 3,433 | 47 | 6.32 | 9.00 |
| Unseen-user holdout | All | Training median | 8,894 | 54 | **6.64** | **9.78** |
| Unseen-user holdout | All | Causal user-history median | 8,894 | 54 | **6.64** | **9.78** |
| Unseen-user holdout | All | Gradient boosting | 8,894 | 54 | 8.15 | 10.06 |

For later sessions from represented users, gradient boosting improves modestly over the already strong causal-history baseline. The unseen-user result is worse than the training-median baseline. Therefore this model is useful only as a limited within-population, history-informed future-session experiment; it does not establish a reliable model for new users.

## Reproduction

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m src.models.session_energy
```

The implementation is [`src/models/session_energy.py`](../src/models/session_energy.py). It reads only the raw Dataset 1 CSV and writes `outputs/ml_metrics.csv` and `outputs/ml_actual_vs_predicted.png`. The companion notebook is `notebooks/06_ml_session_energy.ipynb`.

## Remaining limitations

- The source consists of reported residential sessions from 12 sites, with different observation periods. Results may reflect site and time coverage shifts.
- A random session-row split is deliberately not used because it would mix future and past sessions and repeated users.
- User history contains observed previous energy after a session ends. It is unavailable for new users and requires retained history in any real deployment.
- Delivered energy is not battery energy consumed by driving, battery state of charge, or instantaneous charger power.
- This experiment has not been integrated into the NHTS charging simulator. Any future integration requires an explicit mapping from a predicted recorded residential session to a vehicle parking opportunity.
