"""Paired error diagnostics for the chronological charging benchmark."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL = "gradient_boosting_mae"
HISTORY = "history_median"
GLOBAL = "training_median"
TARGETS = {
    "next_plugin_hours": ("Next recorded plug-in", "h"),
    "connection_hours": ("Connection duration", "h"),
    "energy_kwh": ("Delivered energy", "kWh"),
}
HISTORY_BINS = ["0", "1–4", "5–19", "20–49", "50+"]


def save_diagnostics(root, chronological, seed):
    """Use paired user resampling with a session-weighted MAE estimand."""
    rng = np.random.default_rng(seed)
    comparisons, history_rows = [], []
    for target, errors in chronological.items():
        per_user = errors.groupby("user_id").agg(
            baseline=(HISTORY, "sum"), model=(MODEL, "sum"), sessions=(MODEL, "size")
        )
        draws = rng.integers(len(per_user), size=(5000, len(per_user)))
        totals = per_user.sessions.to_numpy()[draws].sum(axis=1)
        draws_mae = per_user[["baseline", "model"]].to_numpy()[draws].sum(axis=1) / totals[:, None]
        for method, values, estimate in (
            (HISTORY, draws_mae[:, 0], errors[HISTORY].mean()),
            (MODEL, draws_mae[:, 1], errors[MODEL].mean()),
            ("baseline_minus_model", draws_mae[:, 0] - draws_mae[:, 1],
             (errors[HISTORY] - errors[MODEL]).mean()),
        ):
            low, high = np.quantile(values, [.025, .975])
            comparisons.append(dict(target=target, method=method, estimate=estimate,
                                    ci_low=low, ci_high=high, users=len(per_user),
                                    sessions=len(errors), resamples=len(draws), seed=seed))
        bins = pd.cut(errors.history_count, [-1, 0, 4, 19, 49, np.inf], labels=HISTORY_BINS)
        for label, group in errors.groupby(bins, observed=True):
            history_rows.append(dict(target=target, history=label, sessions=len(group),
                                     users=group.user_id.nunique(),
                                     **group[[GLOBAL, HISTORY, MODEL]].mean().to_dict()))

    gaps = chronological["next_plugin_hours"]
    gap_bins = pd.cut(gaps.target, [-1, 24, 72, 168, np.inf],
                      labels=["0–24 h", ">24–72 h", ">72–168 h", ">168 h"])
    gap_rows = []
    for label, group in gaps.groupby(gap_bins, observed=True):
        gain = group[HISTORY] - group[MODEL]
        gap_rows.append(dict(observed_gap=label, sessions=len(group),
                             users=group.user_id.nunique(), session_share=len(group) / len(gaps),
                             **group[[GLOBAL, HISTORY, MODEL]].mean().to_dict(),
                             mean_gain_hours=gain.mean(),
                             contribution_to_overall_gain_hours=gain.sum() / len(gaps)))

    comparison = pd.DataFrame(comparisons)
    history = pd.DataFrame(history_rows)
    tables = root / "outputs/tables/audited_ml"
    for name, table in (("paired_bootstrap", comparison), ("history_length", history),
                        ("next_gap_bins", pd.DataFrame(gap_rows))):
        table.to_csv(tables / f"{name}.csv", index=False, float_format="%.6f")
    figures = root / "outputs/figures/audited_ml"
    figures.mkdir(parents=True, exist_ok=True)
    plot_comparison(comparison, figures)
    plot_history(history, figures)


def plot_comparison(comparison, figures):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for ax, (target, (title, units)) in zip(axes, TARGETS.items()):
        rows = comparison.loc[comparison.target.eq(target)].set_index("method")
        for y, method, color in [(1, HISTORY, "#287a70"), (0, MODEL, "#2559a1")]:
            row = rows.loc[method]
            ax.plot([row.ci_low, row.ci_high], [y, y], color=color, lw=2)
            ax.scatter(row.estimate, y, color=color, zorder=3)
        gain = rows.loc["baseline_minus_model"]
        ax.set(yticks=[0, 1], yticklabels=["Boosting", "History median"], ylim=(-.5, 1.5),
               xlabel=f"Mean absolute error ({units})", title=title)
        ax.text(.5, -.24, f"Paired gain {gain.estimate:.2f} {units}\n"
                f"95% CI [{gain.ci_low:.2f}, {gain.ci_high:.2f}]",
                ha="center", va="top", transform=ax.transAxes)
        ax.grid(axis="x", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model vs. user-history baseline on later Norway sessions\n"
                 "95% intervals from resampling users")
    fig.savefig(figures / "01_paired_uncertainty.png", dpi=180)
    plt.close(fig)


def plot_history(history, figures):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for ax, (target, (title, units)) in zip(axes, TARGETS.items()):
        rows = history.loc[history.target.eq(target)].set_index("history").reindex(HISTORY_BINS)
        x = np.arange(len(rows))
        for method, label, color in [(GLOBAL, "Training median", "#78808c"),
                                      (HISTORY, "History median", "#287a70"),
                                      (MODEL, "Boosting", "#2559a1")]:
            ax.plot(x, rows[method], "o-", label=label, color=color)
        tick_labels = [f"{label}\n({n:,.0f})" if pd.notna(n) else f"{label}\n(0)"
                       for label, n in rows.sessions.items()]
        ax.set(xticks=x, xticklabels=tick_labels, title=title,
               ylabel=f"Mean absolute error ({units})")
        ax.grid(axis="y", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle("Prediction error by available user history\n"
                 "Previous observations (test sessions in parentheses)")
    fig.savefig(figures / "02_history_length.png", dpi=180)
    plt.close(fig)
