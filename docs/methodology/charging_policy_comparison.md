# Immediate versus delayed Home charging

## Research question and reproducibility

How does a common 23:00 charging start affect aggregate EV peak demand and
energy delivered by the assumed next departure?

Run from the repository root:

```powershell
.venv/Scripts/python.exe -m src.charging.compare_policies
```

This produces `outputs/tables/charging_comparison/{summary.csv,profiles.csv,
vehicle_results.csv,cohort_audit.csv,metadata.json}` and two presentation figures
under `outputs/figures/charging_comparison/`. Raw inputs are preserved.

## Scope and assumptions

Input: the existing derived NHTS chains and eligible internal parking gaps.
Select complete multi-leg chains inside the 04:00–04:00 survey window with
the required gaps, and restrict to chains starting and ending at Home.
This restriction gives the assumed next-day Home departure an interpretable
anchor; it is a study cohort, not all NHTS vehicles.

Both policies have the same observed internal Home gaps and the same synthetic
terminal Home interval: final arrival until first departure + 24 hours. No
pre-first-trip charging is added. Next-day departure is assumed, not observed.

- Immediate: start charging at each Home arrival, stopping at departure or target.
- Delayed: begin at 23:00 of the survey day or Home arrival, whichever is later,
  stopping at departure or target. A Home interval ending before 23:00 receives
  no charge. Arrivals after midnight can start immediately. This is a single-day
  scheduling rule, not an optimized controller or a recurring multi-day tariff.
- Both policies assume a 60 kWh battery, 80% initial SoC at 04:00, 90% target,
  0.18 kWh/km and 90% grid-to-battery efficiency. Charger ratings of 3.7, 7.4 and
  11 kW are separate assumed scenarios; all vehicles share the scenario rating.
- A vehicle must be feasible under both policies at all three ratings to enter
  any comparison curve. Each scenario's feasibility and all common-cohort
  exclusions are saved, including reasons. This creates a consistent comparison
  but may exclude travel that would require daytime charging.

## Accounting and interpretation

Each trip deducts its derived energy from the battery. Infeasible trips are
flagged rather than clipped. Each policy is simulated independently, so any
missed daytime charging affects energy available for later trips.

Grid energy is the integral of the 15-minute bin-average power. Unmet target
energy is the battery-side deficit to 90% SoC at synthetic next departure,
summed across the common cohort. It is not unmet driving energy: driving is
feasible for every included vehicle. Target attainment is reported alongside
the peak; lower delivered energy must not be counted as a charging benefit.

The run checks profile/event energy agreement, SoC balance and bounds,
parking constraints and per-vehicle charger limits. Across each policy pair,
the difference in grid energy times efficiency must equal the difference in
final battery deficit. Initial SoC below target means energy includes initial
battery replenishment, not just energy consumed by the day's travel.

Charging after next-day 04:00 is folded onto the representative 24-hour clock,
as in the existing prototype. The curve is not a continuous two-day trajectory
or a proven steady-state battery cycle. The result is unweighted sample demand,
not a national forecast, tariff recommendation or grid power-flow result.

## Presentation figures

- `01_policy_comparison.png`: two demand profiles at 7.4 kW.
- `02_charger_sensitivity.png`: peak and unmet target energy versus charger power.

Interpret a synchronized delayed start as a potential source of a new peak;
do not assume that shifting charging later reduces the maximum demand.

## Results — executed 2026-11-26

Of 6,225 valid chains, 5,702 start and end at Home. The fixed comparison cohort
contains 5,647 vehicle-days feasible in all six scenarios; 55 are excluded.
The per-scenario audit retains their feasibility and failure reasons. No survey
weights or population expansion factors are used.

| Charger (kW) | Immediate peak (MW) | Delayed peak (MW) | Immediate unmet target (kWh battery) | Delayed unmet target (kWh battery) |
|---|---:|---:|---:|---:|
| 3.7 | 8.21 | 20.80 | 229.15 | 1,954.78 |
| 7.4 | 9.78 | 41.25 | 0.32 | 21.97 |
| 11.0 | 10.62 | 61.07 | approximately 0 | approximately 0 |

At 7.4 kW, immediate charging peaks at 18:00–18:15; delayed charging peaks at
23:30–23:45. The peak increases by 321.9% (4.22 times the baseline). Grid energy
is 82,469.33 versus 82,445.27 kWh: delayed charging serves 24.06 kWh less, which
corresponds to 21.65 kWh more battery deficit at 90% efficiency. One immediate
vehicle and ten delayed vehicles miss the final 90% target. Nearly equal
energy served means the large peak increase primarily reflects synchronization.

At 3.7 kW, delayed charging leaves 237 vehicles below target compared with 33
under immediate charging. At 11 kW, both meet the final target for every vehicle,
but the delayed peak is still much larger. Thus the finding persists across
all three ratings and does not depend on unmet target energy.

The original all-feasible prototype peak of 10.14 MW is not the comparator
here: that curve uses 6,136 vehicle-days, whereas these curves use a fixed
5,647-member Home-to-Home cohort.

All six scenario runs completed with energy, battery and parking checks passed.
The original full-cohort validation also reproduced unchanged, and the four
existing terminal-charging checks passed; no extensive new unit-test suite
was introduced.

**Presentation takeaway:** Under these assumptions, moving everyone to one
common late-night start shifts and amplifies the EV-only peak. Delaying alone
does not provide peak management. A future staggered or coordinated policy
should be assessed against this baseline while preserving delivered energy.
No household base load is included, so the result does not establish the
change in total feeder peak, voltage or losses.
