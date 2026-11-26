"""Reproduce and audit the existing prototype without overwriting its figures.

Run from the repository root: python -m src.charging.validate_prototype
"""

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .simple_ev import (
    EVParameters, simulate_vehicle, reconstruct_terminal_home_charge,
    DAY_START_MIN, DAY_END_MIN, BIN_MIN, BIN_STARTS_MIN,
)


def run(root: Path) -> dict:
    params = EVParameters()
    types = {key: "string" for key in
             ("HOUSEID", "VEHCASEID", "TDCASEID", "next_TDCASEID")}
    legs = pd.read_csv(root / "data/interim/vehicle_trip_chains.csv", dtype=types)
    gaps = pd.read_csv(root / "data/interim/vehicle_parking_intervals.csv", dtype=types)
    if not legs.TDCASEID.is_unique:
        raise ValueError("Trip identifiers are not unique")
    vehicles = legs.groupby("VEHCASEID").agg(
        n=("TDCASEID", "size"), status=("chain_status", "first"),
        start=("survey_start_min", "min"), end=("survey_end_min", "max"))
    eligible = gaps.loc[gaps.analysis_eligible.eq(True)]
    counts = eligible.groupby("VEHCASEID").size().reindex(vehicles.index, fill_value=0)
    selected = vehicles.loc[
        vehicles.status.eq("complete") & vehicles.n.ge(2)
        & vehicles.start.ge(DAY_START_MIN) & vehicles.end.le(DAY_END_MIN)
        & counts.eq(vehicles.n - 1)]
    gap_groups = dict(tuple(eligible.groupby("VEHCASEID")))
    rows = []
    baseline_power = np.zeros(len(BIN_STARTS_MIN))
    overnight_power = baseline_power.copy()
    for vehicle_id, group in legs.loc[legs.VEHCASEID.isin(selected.index)].groupby("VEHCASEID"):
        result = simulate_vehicle(group, gap_groups[vehicle_id], params)
        row = {"vehicle_id": vehicle_id, "feasible": result.feasible,
               "failure_reason": result.failure_reason}
        if not result.feasible:
            rows.append(row)
            continue
        terminal = reconstruct_terminal_home_charge(group, result, params)
        events = result.events
        charged = events.loc[events.grid_charge_kwh.gt(0)]
        checks = [
            np.isclose(result.grid_energy_kwh, result.power_kw.sum() * BIN_MIN / 60),
            np.isclose(events.battery_charge_kwh.sum(),
                       result.grid_energy_kwh * params.charging_efficiency),
            charged.location.eq("Home").all(),
            charged.charge_end_min.le(charged.end_min + 1e-8).all(),
            charged.soc_after.le(params.target_soc + 1e-8).all(),
            result.power_kw.min() >= -1e-8,
            result.power_kw.max() <= params.home_charger_grid_kw + 1e-8,
        ]
        target_deficit = max(0., params.battery_capacity_kwh
                             * (params.target_soc - result.soc_fraction[-1]))
        gain = terminal.grid_energy_kwh * params.charging_efficiency
        if terminal.arrival_min is not None:
            available = (terminal.departure_min - terminal.arrival_min) / 60
            expected = min(target_deficit, available * params.home_charger_grid_kw
                           * params.charging_efficiency)
            checks.extend([np.isclose(gain, expected),
                           terminal.charge_end_min <= terminal.departure_min + 1e-8])
        else:
            checks.append(terminal.grid_energy_kwh == 0)
        if not all(checks):
            raise AssertionError(f"Independent accounting check failed: {vehicle_id}")
        baseline_power += result.power_kw
        overnight_power += result.power_kw + terminal.power_kw
        rows.append({**row, "driving_kwh": result.driving_energy_kwh,
                     "internal_grid_kwh": result.grid_energy_kwh,
                     "terminal_grid_kwh": terminal.grid_energy_kwh,
                     "terminal_home_assumed": terminal.arrival_min is not None,
                     "terminal_unmet_target_kwh": max(0., target_deficit - gain)
                     if terminal.arrival_min is not None else np.nan})
    table = pd.DataFrame(rows)
    if table.empty or not table.feasible.any():
        raise ValueError("No feasible vehicle-days")
    summaries = []
    for name, power in [("observed_internal_only", baseline_power),
                        ("synthetic_terminal_home", overnight_power)]:
        peak = int(np.argmax(power))
        minute = int(BIN_STARTS_MIN[peak])
        summaries.append({"scenario": name,
                          "grid_energy_kwh": float(power.sum() * BIN_MIN / 60),
                          "peak_kw": float(power[peak]),
                          "peak_bin_start": f"{minute // 60 % 24:02d}:{minute % 60:02d}",
                          "peak_day_offset": minute // 1440})
    report = {"parameters_assumed": asdict(params),
              "candidate_vehicles": len(vehicles), "selected_vehicle_days": len(selected),
              "feasible_vehicle_days": int(table.feasible.sum()),
              "infeasible_vehicle_days": int((~table.feasible).sum()),
              "failure_counts": table.loc[~table.feasible, "failure_reason"].str.split(": ").str[-1].value_counts().to_dict(),
              "all_accounting_checks_passed": True, "scenarios": summaries}
    out = root / "outputs/tables/prototype_validation"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "vehicle_checks.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "scenario_summary.csv", index=False)
    pd.DataFrame({"survey_minute": BIN_STARTS_MIN,
                  "internal_only_kw": baseline_power,
                  "synthetic_terminal_home_kw": overnight_power}).to_csv(out / "profiles.csv", index=False)
    (out / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(Path(__file__).resolve().parents[2]), indent=2))
