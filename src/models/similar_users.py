"""Test whether similar training users help predict connection duration."""

import json
from pathlib import Path

from .audited_behaviour import RANDOM_SEED, build_model, features, load_targets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


HISTORY_SIZES = (1, 3, 5, 10)
NEIGHBOURS = 5
METHODS = ("calendar_median", "own_median", "boosting", "neighbours", "blended")


def clock(timestamps):
    return timestamps.dt.hour.to_numpy() + timestamps.dt.minute.to_numpy() / 60


def profile(sessions):
    """Use only already completed sessions to characterize a user."""
    arrival = clock(sessions.origin) * (2 * np.pi / 24)
    departure = clock(sessions.available) * (2 * np.pi / 24)
    return np.array([
        np.sin(arrival).mean(), np.cos(arrival).mean(),
        np.sin(departure).mean(), np.cos(departure).mean(),
        np.log1p(sessions.target.median()),
        (sessions.available.dt.normalize() > sessions.origin.dt.normalize()).mean(),
        sessions.origin.dt.dayofweek.ge(5).mean(),
    ])


def pilot_split(root):
    targets, _, cutoff = load_targets(root / "data/raw/norway_ev/Dataset1_charging_reports.csv")
    sessions = targets["connection_hours"]
    users = sorted(sessions.user_id.unique())
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=RANDOM_SEED)
    _, held = next(splitter.split(users, groups=users))
    held_mask = sessions.user_id.isin({users[i] for i in held})
    train = sessions.loc[~held_mask & sessions.available.lt(cutoff)]
    test = sessions.loc[held_mask & sessions.origin.ge(cutoff)]
    return train, test, cutoff, len(held)


def checkpoints(sessions):
    """Yield the first forecast per user at each exact completion count."""
    for user, group in sessions.groupby("user_id", sort=True):
        completed = group.sort_values(["available", "session_id"])
        available = pd.DatetimeIndex(completed.available)
        used = set()
        for session in group.sort_values(["origin", "session_id"]).itertuples():
            n = int(available.searchsorted(session.origin, side="left"))
            if n in HISTORY_SIZES and n not in used:
                used.add(n)
                yield user, session, n, completed.iloc[:n]


def benchmark(train, test):
    references = {
        user: group.sort_values(["available", "session_id"])
        for user, group in train.groupby("user_id", sort=True)
    }
    # Use the same history length on each side of the distance calculation.
    banks = {}
    for n in HISTORY_SIZES:
        ids = [user for user, group in references.items() if len(group) >= n]
        matrix = np.stack([profile(references[user].tail(n)) for user in ids])
        # Fixed scales keep the two coordinates of each clock feature comparable.
        scale = np.array([1., 1., 1., 1., .75, .5, .5])
        banks[n] = (np.array(ids), matrix / scale, scale)

    test_x = features(test, test)
    model = build_model().set_params(model__loss="absolute_error", model__early_stopping=False)
    model.fit(features(train, train), train.target)
    boosting = pd.Series(model.predict(test_x), index=test_x.index)
    calendar = train.groupby([train.origin.dt.dayofweek, train.origin.dt.hour]).target.median()
    median = float(train.target.median())
    predictions = []
    for user, session, n, past in checkpoints(test):
        assert past.available.lt(session.origin).all()
        assert test_x.at[session.Index, "history_count"] == n
        ids, matrix, scale = banks[n]
        distances = np.linalg.norm(matrix - profile(past) / scale, axis=1)
        indices = np.argsort(distances, kind="stable")[:NEIGHBOURS]
        weights = 1 / (distances[indices] + .25)
        hour = session.origin.hour + session.origin.minute / 60
        donor_predictions = []
        for donor in ids[indices]:
            group_train = references[donor]
            deltas = np.abs((clock(group_train.origin) - hour + 12) % 24 - 12)
            same_hour = group_train.loc[deltas <= 3, "target"]
            donor_predictions.append((same_hour if len(same_hour) else group_train.target).median())
        neighbour = float(np.average(donor_predictions, weights=weights))
        own = float(past.target.median())
        own_weight = n / (n + 5)
        predictions.append(dict(user_id=user, history=n, actual_hours=session.target,
                                calendar_median=calendar.get((session.origin.dayofweek, session.origin.hour), median),
                                own_median=own, boosting=boosting.at[session.Index],
                                neighbours=neighbour,
                                blended=own_weight * own + (1 - own_weight) * neighbour))

    predictions = pd.DataFrame(predictions)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for n, cohort in predictions.groupby("history"):
        errors = cohort[list(METHODS)].sub(cohort.actual_hours, axis=0).abs()
        draws = rng.integers(len(cohort), size=(5000, len(cohort)))
        for method in METHODS:
            gain = (errors.boosting - errors[method]).to_numpy()
            low, high = np.quantile(gain[draws].mean(axis=1), [.025, .975])
            rows.append(dict(history=n, users=len(cohort), method=method,
                             mae_hours=errors[method].mean(), gain_vs_boosting_hours=gain.mean(),
                             gain_ci_low=low, gain_ci_high=high))
    metrics = pd.DataFrame(rows)
    return metrics


def run(root):
    train, test, cutoff, held_users = pilot_split(root)
    metrics = benchmark(train, test)
    out = root / "outputs/tables/similar_users"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.6f")
    (out / "metadata.json").write_text(json.dumps(dict(
        cutoff=str(cutoff), train_users=int(train.user_id.nunique()), train_sessions=len(train),
        held_users=held_users,
        post_cutoff_held_users=int(test.user_id.nunique()), seed=RANDOM_SEED,
        neighbours=NEIGHBOURS), indent=2), encoding="utf-8")

    figure_dir = root / "outputs/figures/audited_ml"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    labels = ["Calendar median", "Own history median", "Gradient boosting",
              "Similar users", "Similar users + own history"]
    for method, label in zip(METHODS, labels):
        scores = metrics.loc[metrics.method.eq(method)]
        ax.plot(scores.history, scores.mae_hours, "o-", label=label)
    counts = metrics.groupby("history").users.first()
    ax.set(xticks=counts.index, xticklabels=[f"{n}\n({count} users)" for n, count in counts.items()],
           xlabel="Completed sessions known at prediction time", ylabel="Mean absolute error (h)",
           title="Connection duration for held-out users with limited history")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "03_similar_users.png", dpi=180)
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(run(Path(__file__).resolve().parents[2]).to_string(index=False))
