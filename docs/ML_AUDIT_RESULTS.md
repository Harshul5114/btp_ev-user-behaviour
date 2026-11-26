# Charging behaviour prediction audit

Run from the repository root:

```powershell
.venv/Scripts/python.exe -m src.models.audited_behaviour
```

The command writes `outputs/tables/audited_ml/metrics.csv` and `metadata.json`.
It uses reported Norway sessions, calendar information and user history.
The same gradient boosting family as the prior energy model is fitted with
absolute-error loss and fixed parameters. No NHTS priors or synthetic data
enter this benchmark.

## Findings from the previous code

- The energy pipeline registered labels at plug-in and seeded the chronological
  test with all training-period labels, including sessions still connected at
  the cutoff. The legacy rerun reproduced the saved CSV values, including
  5.612 kWh overall chronological MAE and 8.231 kWh zero-history MAE for
  boosting. Several values in the previous prose differ from that CSV.
- The timing pipeline builds next gaps after filtering out incomplete sessions,
  clips overlaps to zero, and records a gap in history before the next plug-in
  reveals it. Its split uses current plug-in for both targets, with no check
  that the training label is known by the cutoff. Shuffled folds are used in
  the site encoding and the overnight classifier. Prior timing scores should
  be treated as provisional. The legacy command also fails in the current
  environment because its undeclared `holidays` dependency is missing.
- The audited command avoids those legacy preprocessing paths. It builds
  next-event labels from successive recorded plug-ins, including successors
  with missing plug-outs. Energy and duration enter history after recorded
  plug-out. Next-gap labels enter history after the next plug-in. Events with
  the same timestamp as a prediction are conservatively withheld.

## Evaluation

The chronological cutoff is 2020-01-26 00:16:34. The origin is plug-out for
next plug-in prediction and plug-in for connection duration and energy.
Training labels must be available strictly before the cutoff. This removes
134 next-gap labels and 51 connection/energy labels per target from training.
The chronological benchmark updates history as observations arrive. The
separate user holdout uses 54 users and gives them no history. It isolates
user generalization across the study period.

Of 35,377 source sessions, 38 lack plug-out and are excluded as starting
records for all three targets. The gap evaluation also excludes 70 overlaps
and conditions on a recorded successor. There are 267 last records without
a known successor. These represent censored or incomplete follow-up, and
the results apply to recorded sessions with valid completed gaps.

| Target | Chronological sessions | Best baseline MAE | Boosting MAE | Zero-history user holdout baseline / boosting MAE |
|---|---:|---:|---:|---:|
| Next recorded plug-in | 6,836 | 46.03 h (history) | 44.61 h | 28.05 / 31.38 h |
| Connection duration | 7,071 | 8.36 h (history) | 7.78 h | 7.83 / 8.50 h |
| Session energy | 7,071 | 5.64 kWh (history) | 5.52 kWh | 6.64 / 6.80 kWh |

The next-plug-in model reduces overall chronological MAE by 3.1%, but for
users seen in training it scores 36.79 h against a 36.11 h history median.
Across the full chronological test it places 61.2% of events within 24 hours
of the true timestamp, versus 63.4% for the training median. Connection
duration shows the clearest overall improvement, from 8.36 h to 7.78 h, and
the model is within six hours for 66.1% of events versus 57.5% for the user
history median. The zero-history holdout favours simple baselines by MAE
for all three targets.

These are point estimates from a reused historical evaluation period. The
benchmark does not retune hyperparameters or validate the previous quantile
intervals. Use paired uncertainty estimates and history-length diagnostics
before making claims about the reliability of small differences.
