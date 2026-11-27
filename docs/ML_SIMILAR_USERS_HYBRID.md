# Similar-user features in connection-duration boosting

This follow-up compares learned use of similar drivers with the earlier
fixed-weight approach. It reuses the first prediction at each history size
for the exact held-out sessions in [the original pilot](ML_SIMILAR_USERS.md).

```powershell
.venv/Scripts/python.exe -m src.models.similar_users_hybrid
```

The personal features summarize arrival and departure clock times, overnight
and weekend tendencies, recent durations, spread in durations, and time since
the last plug-out. The similar-user features add a predicted duration, the
closest match distance, disagreement among matches, and the number of donors.
Training donors come from other training users who had completed sessions
before the first day of the forecast month. Test donors are frozen at the
January 2020 cutoff. The held-out driver's profile uses only sessions completed
before the forecast.

| Completed sessions | Held-out users | Original boosting | Boosting on checkpoints | Rich personal history | Personal history + similar users | Previous fixed blend |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38 | 8.64 h | 9.66 h | 9.56 h | 9.69 h | 8.92 h |
| 3 | 31 | 7.88 h | 7.91 h | 8.44 h | 8.39 h | 7.75 h |
| 5 | 23 | 3.93 h | 5.88 h | 5.36 h | 5.73 h | 4.21 h |
| 10 | 20 | 3.78 h | 5.54 h | 5.23 h | 5.38 h | 4.75 h |

All values are MAE for connection duration. The matched feature comparison
uses 682 training checkpoints for each of the three restricted models, 624
with at least one eligible donor. The original benchmark uses 21,168 training
sessions. That difference limits conclusions about richer features at larger
training sizes.

Adding donor features to rich personal history improves the point estimate only
at three sessions, by 0.05 h. All paired 95% user-bootstrap intervals for this
feature addition include zero. The original full-data boosting model has lower
MAE than the learned hybrid at every checkpoint, and the earlier fixed blend
also has lower MAE than the learned hybrid at every checkpoint. Retain the
original boosting model as the benchmark and treat the neighbourhood methods
as exploratory. Cohort membership varies by history size, so comparisons
between methods should be made within a row.

See `outputs/tables/similar_users_hybrid/` for metrics and metadata, and
`outputs/figures/audited_ml/04_similar_users_hybrid.png` for the comparison.
