# Charging prototype validation — 2026-11-26

The existing NHTS charging baseline was reproduced from the interim files.
This validates the charging stage, not the entire notebook or ML pipeline.
Raw data and previous figures were preserved.

## Reproduce

From the repository root:

```powershell
.venv/Scripts/python.exe -m src.charging.validate_prototype
.venv/Scripts/python.exe -m unittest discover -s tests -p test_terminal_charging.py -v
```

The validation command writes per-vehicle checks, scenario summaries, 15-minute
profiles and assumed parameters to `outputs/tables/prototype_validation/`.
No random sampling is used.

## Verified results

- 6,874 candidate vehicles; 6,225 selected multi-leg, complete, in-window chains
  with the required eligible internal gaps, matching notebook 05's selection.
- 6,136 feasible vehicle-days contribute to both aggregates.
- 89 excluded days: 86 exceed available battery energy during a trip; three
  report positive distance with zero travel duration. Excluded days remain in
  the audit CSV; they are not silently dropped from the reported counts.
- All feasible cases passed the simulator's battery balance, SoC and integrated
  power checks, plus independent checks of charging location, efficiency,
  departure limits, target SoC, charger rating and terminal available energy.
- Four synthetic known-answer tests passed: charging across next-day 04:00,
  departure-limited charging with efficiency, a non-Home final destination,
  and a battery already at target.

| Scenario | Grid energy (kWh) | Peak (kW) | Peak 15-minute bin |
|---|---:|---:|---|
| Observed internal Home gaps only | 10,767.9 | 1,240.8 | 12:45–13:00 |
| Internal gaps plus synthetic terminal Home parking | 86,606.0 | 10,138.1 | 18:00–18:15 |

These reproduce the earlier baseline figures. The difference demonstrates
sensitivity to the missing terminal parking window; it is **not** an
immediate-versus-delayed policy comparison because the available charging
windows and energy served differ.

## Interpretation and limits for the presentation

NHTS supplies reported trips and derived internal parking gaps. EV parameters
are assumptions: 60 kWh capacity, 0.18 kWh/km consumption, 7.4 kW grid-side
charger, 90% efficiency, 80% initial SoC and 90% target SoC. Terminal Home
parking ends at the same vehicle-day's first departure clock time the next day.
That departure is synthetic. No synthetic trips or users were added.

The overnight curve folds charging after next-day 04:00 onto a representative
24-hour clock. It is not a continuous multi-day simulation or a verified cyclic
SoC trajectory. Initial SoC is below target, so served energy includes the
initial-to-target battery increment as well as travel replenishment. Demand is
an unweighted sample scenario, not a population forecast or a grid result.

## Next experiment

Compare immediate and delayed charging on identical parking opportunities and
the same vehicle cohort. Specify the delayed policy, deadline and overnight
boundary before implementing it. Report served energy and unmet target energy
alongside peak reduction; do not count unserved energy as a policy benefit.
Revisit feasibility for both policies and report exclusions explicitly.
