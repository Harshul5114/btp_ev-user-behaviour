# Electric Nation dataset intake

The [Electric Nation customer trial report](https://eatechnology.com/media/girhcnsc/electric-nation-customer-trial-report.pdf)
describes transaction records from CrowdCharge and GreenFlux with a charger ID,
connection and disconnection times in UTC, and energy transferred. The report
describes a separate mapping from charger to participant and registered vehicle.
Use charger history as a proxy for household charging unless that mapping is
available and validates the identity assumption.

The [publisher's Electric Nation data page](https://commercial.nationalgrid.co.uk/electric-nation-data)
lists both transaction exports. The page returned HTTP 403 during acquisition.
The [current catalog API](https://connecteddata.nationalgrid.co.uk/api/3/action/package_show?id=a33247f4-e48e-49f1-ae4e-75e8448bb31c)
was accessible but listed meter archives, survey results and documentation.
Both the CrowdCharge transaction workbook and the meter archive were later
supplied locally. The benchmark uses the workbook's Transaction Data sheet.

## Workbook audit

`TransactionID` is unique across 71,264 rows. `ParticipantID` identifies
300 participants and `ChargerID` identifies 301 chargers. Each known
participant maps to a single charger and vice versa in this export.
`StartTime` and `StopTime` are UTC, while `AdjustedStartTime` and
`AdjustedStopTime` are supplied local clocks. Deriving Europe/London
clocks from UTC matches the adjusted fields for all but one row. Computing
duration from UTC preserves elapsed time across daylight-saving changes.
`ConsumedkWh` is reported energy. Active charging fields are separate
from the plug-in and unplug timestamps.

The benchmark excludes, in order, 3,899 rows without participant IDs,
1,460 with missing or outside 0.5–100 kWh energy, 449 with invalid or under
15-minute durations, 67 longer than seven days and 263 overlapping rows.
The seven-day cap is an analysis choice for unusually long records. The
energy bounds follow the trial report. The cohort contains 65,126 usable
transactions from 299 participants, with median duration 11.58 hours.

## Independent UK benchmark

```powershell
.venv/Scripts/python.exe -m src.models.electric_nation
```

Training uses 42,034 sessions from 239 participants before 28 September
2018. From the 60 disjoint held-out participants, 49 have later sessions.
The benchmark reveals only later completed sessions from each held-out
participant and forecasts the first eligible session at each exact history
size. Each row compares methods on the same forecasts. Metrics and counts
are saved under `outputs/tables/electric_nation/`.

| Completed sessions | Participants | Calendar median MAE | Own median MAE | Boosting MAE | Similar-user blend MAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 6.52 h | 8.01 h | 5.23 h | 6.30 h |
| 3 | 46 | 5.84 h | 7.43 h | 5.94 h | 7.02 h |
| 5 | 45 | 4.99 h | 6.71 h | 4.53 h | 6.03 h |
| 10 | 42 | 5.26 h | 5.22 h | 4.61 h | 5.16 h |

After one completed session, the blend lowers MAE by 1.71 hours relative
to the participant's own median. Boosting has lower MAE than the blend
at all four checkpoints. The calendar median edges out boosting at three
sessions. The test includes 1,125 Trial 3 sessions and the training set
contains no Trial 3 rows, so the split also reflects a change in the trial
mix. Results refer to this CrowdCharge cohort, with its interventions
and observation coverage. Norway scores are reported separately.
