"""Generate calibrated datasets and compare connection-duration predictors."""

import argparse
import json
from pathlib import Path

from .similar_users import benchmark
from .similar_users_hybrid import benchmark_hybrid
from src.data.calibrated_charging import CUTOFF, SCENARIOS, calibration, generate

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY = {
    "calendar_median": "Calendar median", "own_median": "Own median",
    "boosting": "Original boosting", "neighbours": "Similar users",
    "blended": "Fixed blend", "checkpoint_boosting": "Checkpoint boosting",
    "personal_history": "Richer personal history",
    "personal_plus_donors": "Personal + similar users",
}


def run(root, users, sessions, seeds):
    references, calibration_details = calibration(root)
    print("Calibration complete", calibration_details, flush=True)
    raw = root / "data/synthetic/calibrated_charging"
    out = root / "outputs/tables/synthetic_benchmark"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    results, details = [], []
    for scenario in SCENARIOS:
        for seed in seeds:
            synthetic = generate(references, scenario, seed, users, sessions)
            assert len(synthetic) == users * sessions and synthetic.session_id.is_unique
            assert synthetic.target.gt(0).all()
            previous_end = synthetic.groupby("user_id").available.shift()
            assert not synthetic.origin.lt(previous_end).any()
            train = synthetic.loc[synthetic.cohort.eq("train")]
            test = synthetic.loc[synthetic.cohort.eq("test")]
            assert train.available.lt(CUTOFF).all() and test.origin.ge(CUTOFF).all()
            assert not set(train.user_id) & set(test.user_id)
            filename = f"{scenario}_{seed}_{users}users.csv.gz"
            synthetic.to_csv(raw / filename, index=False, compression="gzip")
            scores = benchmark(train, test)
            hybrid, metadata = benchmark_hybrid(train, test, CUTOFF)
            basic = scores.loc[scores.method.eq("boosting"), ["history", "users", "mae_hours"]]
            duplicate = hybrid.loc[hybrid.method.eq("original_boosting"), basic.columns]
            assert basic[["history", "users"]].to_numpy().tolist() == duplicate[
                ["history", "users"]].to_numpy().tolist()
            np.testing.assert_allclose(basic.mae_hours, duplicate.mae_hours, atol=1e-10)
            assert basic.users.eq(users // 5).all()
            hybrid = hybrid.loc[~hybrid.method.eq("original_boosting")].rename(columns={
                "gain_vs_original_boosting": "gain_vs_boosting_hours",
                "original_boosting_ci_low": "gain_ci_low", "original_boosting_ci_high": "gain_ci_high",
            })
            combined = pd.concat([scores, hybrid], ignore_index=True).assign(scenario=scenario, seed=seed)
            combined["rank"] = combined.groupby("history").mae_hours.rank(method="min")
            results.append(combined)
            metadata["bootstrap_seed"] = metadata.pop("seed")
            details.append(dict(scenario=scenario, seed=seed, file=filename, sessions=len(synthetic),
                                users=users, median_duration_hours=float(synthetic.target.median()),
                                median_gap_days=float((synthetic.origin - previous_end).dt.total_seconds().median() / 86400),
                                median_user_duration_sd_hours=float(synthetic.groupby("user_id").target.std().median()),
                                **metadata))
            pd.concat(results, ignore_index=True).to_csv(out / "per_run.csv", index=False, float_format="%.6f")
            print(f"Completed {scenario}, seed {seed}: {len(synthetic):,} sessions, "
                  f"{metadata['test_checkpoints']} matched forecasts", flush=True)
    scores = pd.concat(results, ignore_index=True)
    rankings = scores.assign(win=scores["rank"].eq(1)).groupby(
        ["scenario", "history", "method"], as_index=False
    ).agg(mean_mae_hours=("mae_hours", "mean"), min_seed_mae=("mae_hours", "min"),
          max_seed_mae=("mae_hours", "max"), mean_rank=("rank", "mean"),
          wins=("win", "sum"), mean_gain_vs_boosting_hours=("gain_vs_boosting_hours", "mean"),
          seeds=("seed", "size"), users_per_seed=("users", "min"))
    rankings = rankings.sort_values(["scenario", "history", "mean_mae_hours"])
    rankings.to_csv(out / "rankings.csv", index=False, float_format="%.6f")
    (out / "metadata.json").write_text(json.dumps(dict(
        calibration=calibration_details, scenario_weights=SCENARIOS, seeds=seeds, datasets=details,
    ), indent=2), encoding="utf-8")
    plot(root, rankings)
    print(rankings.groupby(["scenario", "history"]).head(1).to_string(index=False), flush=True)


def plot(root, rankings):
    chosen = ["own_median", "boosting", "blended", "personal_history",
              "personal_plus_donors", "calendar_median"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), sharey=True, layout="constrained")
    for ax, (scenario, name) in zip(axes, zip(SCENARIOS, ["Shared", "Individual", "Irregular"])):
        for method in chosen:
            values = rankings.loc[(rankings.scenario == scenario) & (rankings.method == method)]
            ax.plot(values.history, values.mean_mae_hours, "o-", label=DISPLAY[method])
        ax.set(title=f"{name} routines", xlabel="Completed sessions", xticks=[1, 3, 5, 10])
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Mean absolute error (hours)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Synthetic connection-duration benchmark across routine scenarios")
    folder = root / "outputs/figures/audited_ml"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "05_synthetic_benchmark.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--sessions", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if args.users < 100 or args.sessions < 11 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use at least 100 users and 11 sessions, and unique seeds.")
    run(Path(__file__).resolve().parents[2], args.users, args.sessions, args.seeds)
