"""Fixed-cohort Home-charging policy comparison and charger sensitivity.

Run: python -m src.charging.compare_policies
"""
from dataclasses import asdict
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".tmp/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .simple_ev import (EVParameters, simulate_vehicle, reconstruct_terminal_home_charge,
                        BIN_STARTS_MIN, BIN_MIN, DAY_START_MIN, DAY_END_MIN)


def run(root: Path) -> pd.DataFrame:
    types = {c: "string" for c in ("VEHCASEID", "TDCASEID", "next_TDCASEID")}
    legs = pd.read_csv(root / "data/interim/vehicle_trip_chains.csv", dtype=types)
    gaps = pd.read_csv(root / "data/interim/vehicle_parking_intervals.csv", dtype=types)
    vehicles = legs.groupby("VEHCASEID").agg(
        n=("TDCASEID", "size"), status=("chain_status", "first"),
        start=("survey_start_min", "min"), end=("survey_end_min", "max"),
        begins_home=("begins_home", "first"), ends_home=("ends_home", "first"))
    eligible = gaps.loc[gaps.analysis_eligible.eq(True)]
    counts = eligible.groupby("VEHCASEID").size().reindex(vehicles.index, fill_value=0)
    valid = vehicles.status.eq("complete") & vehicles.n.ge(2) & vehicles.start.ge(DAY_START_MIN) & vehicles.end.le(DAY_END_MIN) & counts.eq(vehicles.n - 1)
    selected = vehicles.loc[valid & vehicles.begins_home & vehicles.ends_home]
    leg_groups = dict(tuple(legs.loc[legs.VEHCASEID.isin(selected.index)].groupby("VEHCASEID")))
    gap_groups = dict(tuple(eligible.loc[eligible.VEHCASEID.isin(selected.index)].groupby("VEHCASEID")))
    results, audits = {}, []
    common = set(selected.index)
    for rating in (3.7, 7.4, 11.0):
        params = EVParameters(home_charger_grid_kw=rating)
        for policy, threshold in (("immediate", None), ("delayed_2300", 1380.)):
            scenario = {}
            for vehicle_id, chain in leg_groups.items():
                result = simulate_vehicle(chain, gap_groups[vehicle_id], params,
                                          home_charge_not_before_min=threshold)
                audits.append({"charger_kw": rating, "policy": policy,
                               "vehicle_id": vehicle_id, "feasible": result.feasible,
                               "failure_reason": result.failure_reason})
                if not result.feasible:
                    continue
                terminal = reconstruct_terminal_home_charge(
                    chain, result, params, home_charge_not_before_min=threshold)
                power = result.power_kw + terminal.power_kw
                grid = result.grid_energy_kwh + terminal.grid_energy_kwh
                final_battery = (params.initial_soc * params.battery_capacity_kwh
                                 - result.driving_energy_kwh + grid * params.charging_efficiency)
                deficit = max(0., params.target_soc * params.battery_capacity_kwh - final_battery)
                # Independent scenario checks, beyond the simulator's event checks.
                if not np.isclose(grid, power.sum() * BIN_MIN / 60, atol=1e-8):
                    raise AssertionError("Integrated grid energy mismatch")
                if not -1e-8 <= final_battery <= params.battery_capacity_kwh + 1e-8:
                    raise AssertionError("Battery bounds violated")
                if power.min() < -1e-8 or power.max() > rating + 1e-8:
                    raise AssertionError("Per-vehicle charger limit violated")
                charged = result.events.loc[result.events.grid_charge_kwh.gt(0)]
                if not (charged.charge_start_min.ge(charged.start_min - 1e-8).all()
                        and charged.charge_end_min.le(charged.end_min + 1e-8).all()):
                    raise AssertionError("Charging outside parking window")
                if threshold is not None and not charged.charge_start_min.ge(threshold).all():
                    raise AssertionError("Delayed policy started early")
                scenario[vehicle_id] = (power, {
                    "vehicle_id": vehicle_id, "charger_kw": rating, "policy": policy,
                    "grid_energy_kwh": grid, "driving_energy_kwh": result.driving_energy_kwh,
                    "unmet_target_battery_kwh": deficit,
                    "synthetic_departure_min": terminal.departure_min,
                    "final_soc": final_battery / params.battery_capacity_kwh})
            common.intersection_update(scenario)
            results[(rating, policy)] = scenario
            print(f"{rating:g} kW {policy}: {len(scenario)} feasible of {len(selected)}", flush=True)
    if not common:
        raise ValueError("No common feasible cohort")
    common = sorted(common)
    summaries, vehicle_rows, profiles = [], [], {}
    for (rating, policy), scenario in results.items():
        power = np.sum([scenario[v][0] for v in common], axis=0)
        rows = pd.DataFrame([scenario[v][1] for v in common])
        vehicle_rows.extend(rows.to_dict("records"))
        profiles[(rating, policy)] = power
        minute = int(BIN_STARTS_MIN[np.argmax(power)])
        unmet = rows.unmet_target_battery_kwh
        summaries.append({"charger_kw": rating, "policy": policy,
                          "vehicle_days": len(common),
                          "grid_energy_kwh": rows.grid_energy_kwh.sum(),
                          "peak_kw": power.max(),
                          "peak_time": f"{minute // 60 % 24:02d}:{minute % 60:02d}",
                          "unmet_target_battery_kwh": unmet.sum(),
                          "vehicles_below_target": int(unmet.gt(1e-6).sum()),
                          "target_met_pct": 100 * unmet.le(1e-6).mean()})
    summary = pd.DataFrame(summaries)
    for rating in (3.7, 7.4, 11.):
        immediate = summary.loc[(summary.charger_kw == rating) & (summary.policy == "immediate")].iloc[0]
        mask = summary.charger_kw.eq(rating)
        summary.loc[mask, "peak_reduction_vs_immediate_pct"] = 100 * (1 - summary.loc[mask, "peak_kw"] / immediate.peak_kw)
        pair = summary.loc[mask]
        # Same cohort, initial state, trips and target: energy not served must
        # equal the additional final battery deficit (including efficiency).
        balance = (pair.iloc[0].grid_energy_kwh - pair.iloc[1].grid_energy_kwh) * .9
        if not np.isclose(balance, pair.iloc[1].unmet_target_battery_kwh - pair.iloc[0].unmet_target_battery_kwh, atol=1e-6):
            raise AssertionError("Policy energy/deficit accounting mismatch")
    out = root / "outputs/tables/charging_comparison"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(vehicle_rows).to_csv(out / "vehicle_results.csv", index=False)
    audit = pd.DataFrame(audits)
    audit["in_common_cohort"] = audit.vehicle_id.isin(common)
    audit.to_csv(out / "cohort_audit.csv", index=False)
    pd.DataFrame({"survey_minute": BIN_STARTS_MIN, **{
        f"{rating:g}kw_{policy}": power for (rating, policy), power in profiles.items()
    }}).to_csv(out / "profiles.csv", index=False)
    metadata = {"parameters_assumed": asdict(EVParameters()), "charger_ratings_kw": [3.7, 7.4, 11.],
                "all_candidate_vehicles": len(vehicles), "valid_chains": int(valid.sum()),
                "home_to_home_selected": len(selected), "common_feasible_cohort": len(common),
                "excluded_from_comparison": len(selected) - len(common),
                "delayed_threshold_survey_min": 1380,
                "terminal_parking_assumption": "Final Home arrival to first departure + 1440 minutes",
                "checks_passed": True}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot(root, profiles, summary, len(common))
    print(summary.to_string(index=False), flush=True)
    return summary


def plot(root, profiles, summary, count):
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    out = root / "outputs/figures/charging_comparison"
    out.mkdir(parents=True, exist_ok=True)
    colors = {"immediate": "#2563a6", "delayed_2300": "#d46b24"}
    labels = {"immediate": "Immediate Home charging", "delayed_2300": "Home charging from 23:00"}
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.r_[BIN_STARTS_MIN, DAY_END_MIN] / 60
    for policy in colors:
        power = profiles[(7.4, policy)] / 1000
        ax.step(x, np.r_[power, power[-1]], where="post", color=colors[policy], lw=2.5, label=labels[policy])
    ax.set(title=f"A common 23:00 start can concentrate EV charging demand\n{count:,} sample vehicle-days | assumed 7.4 kW Home chargers",
           ylabel="Aggregate grid-side demand (MW)", xlabel="Representative clock time (15-minute averages)", xlim=(4, 28))
    ticks = np.arange(4, 29, 3)
    ax.set_xticks(ticks, [f"{h % 24:02d}:00" + (" +1" if h >= 24 else "") for h in ticks])
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.text(.1, .015, "Unweighted scenario; synthetic next departure repeats first-departure clock time. Post-04:00 charging is folded back.", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(out / "01_policy_comparison.png", dpi=190)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for policy in colors:
        rows = summary.loc[summary.policy.eq(policy)]
        axes[0].plot(rows.charger_kw, rows.peak_kw / 1000, "o-", color=colors[policy], label=labels[policy])
        axes[1].plot(rows.charger_kw, rows.unmet_target_battery_kwh, "o-", color=colors[policy])
    axes[0].set(ylabel="Peak grid-side demand (MW)", title="Aggregate peak")
    axes[1].set(ylabel="Unmet battery energy to 90% target (kWh)", title="Shortfall at assumed next departure")
    for ax in axes:
        ax.set_xlabel("Assumed Home charger rating (kW)")
        ax.set_xticks([3.7, 7.4, 11])
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle(f"Charger sensitivity on the same {count:,} vehicle-days")
    fig.tight_layout()
    fig.savefig(out / "02_charger_sensitivity.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
