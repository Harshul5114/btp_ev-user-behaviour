# Similar-user prediction with limited charging history

Can charging patterns from other drivers improve the prediction of a new
driver's connection duration? This pilot uses the same Norway source and
gradient boosting model family as the audited benchmark. It uses a stricter
split that holds out both users and later dates.

```powershell
.venv/Scripts/python.exe -m src.models.similar_users
```

The reference bank uses 21,168 completed sessions from 179 users before
26 January 2020. Of 54 held-out users, 45 have eligible sessions after the
cutoff. Treat each as enrolling at the cutoff and reveal only their later
completed sessions. For each history size, take the first eligible forecast
per user with exactly that many outcomes available. All methods receive the
same forecast sessions. The current session's duration cannot enter its own
profile. Different users qualify for each history size.

Match donor profiles using circular plug-in and plug-out summaries, median
connection duration, overnight share and weekend share. Build each donor's
profile from the same number of completed sessions as the current user's
profile, using the donor's most recent sessions before the cutoff. Use the
five nearest donors with inverse distance weights. Each contributes its
historical median at plug-in times within three hours of the current plug-in,
or its overall median when that window is empty. The blended method weights
the user's own median by `n / (n + 5)`, where `n` is completed history size.
These settings were fixed for the pilot.

| Prior sessions | Users | Calendar median MAE | Own median MAE | Boosting MAE | Similar users MAE | Blended MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38 | 10.06 h | 11.34 h | 8.64 h | 9.01 h | 8.92 h |
| 3 | 31 | 7.86 h | 8.01 h | 7.88 h | 8.06 h | 7.75 h |
| 5 | 23 | 6.72 h | 4.65 h | 3.93 h | 4.81 h | 4.21 h |
| 10 | 20 | 5.95 h | 4.67 h | 3.78 h | 4.99 h | 4.75 h |

After one completed session, the blend's MAE is 2.42 h lower than the
user's own median and 0.28 h higher than boosting. With three sessions,
the blend is 0.13 h ahead of boosting, with a paired 95% bootstrap interval
of −0.88 to 1.50 h for the gain. Boosting has the lowest point estimate at
one, five and ten sessions. All four comparison intervals for the blend
versus boosting cross zero. With only 20–38 users per comparison, this is
early evidence that borrowing information can stabilize a personal median
when history is sparse. It gives little evidence of a gain over boosting.

The run saves metrics and split metadata under `outputs/tables/similar_users/`
and a comparison figure at `outputs/figures/audited_ml/03_similar_users.png`.
