"""Compare richer personal history and donor features in duration boosting."""

from functools import lru_cache
import json
from pathlib import Path

from .audited_behaviour import RANDOM_SEED, build_model, features
from .similar_users import NEIGHBOURS, checkpoints, clock, pilot_split, profile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROFILE_NAMES = ["arrival_sin", "arrival_cos", "departure_sin", "departure_cos",
                 "log_median_duration", "overnight_share", "weekend_share"]
SCALE = np.array([1., 1., 1., 1., .75, .5, .5])
DONOR_NAMES = ["donor_estimate", "donor_distance", "donor_disagreement", "donor_count"]


def checkpoint_features(records, base, bank_at, cutoff=None):
    rows = []
    for user, session, n, past in checkpoints(records):
        assert past.available.lt(session.origin).all()
        assert base.at[session.Index, "history_count"] == n
        # A monthly snapshot makes every training donor label strictly historical.
        timestamp = cutoff if cutoff is not None else session.origin.normalize().replace(day=1)
        references, users, profiles = bank_at(timestamp, n)
        assert timestamp <= session.origin
        personal = profile(past)
        distances = np.linalg.norm(profiles - personal / SCALE, axis=1)
        distances[users == user] = np.inf
        valid = np.flatnonzero(np.isfinite(distances))
        nearest = valid[np.argsort(distances[valid], kind="stable")[:NEIGHBOURS]]
        hour = session.origin.hour + session.origin.minute / 60
        estimates = []
        for donor in users[nearest]:
            donor_hours, duration = references[donor]
            close = np.abs((donor_hours - hour + 12) % 24 - 12) <= 3
            estimates.append(float(np.median(duration[close] if close.any() else duration)))
        if estimates:
            weight = 1 / (distances[nearest] + .25)
            estimate = np.average(estimates, weights=weight)
            disagreement = np.sqrt(np.average((np.asarray(estimates) - estimate) ** 2,
                                               weights=weight))
            donor_values = [estimate, distances[nearest[0]], disagreement, len(nearest)]
        else:
            donor_values = [np.nan, np.nan, np.nan, 0]
        rows.append({
            "index": session.Index, "user": user, "history": n,
            **dict(zip(PROFILE_NAMES, personal)),
            "last_duration": past.target.iloc[-1],
            "recent_median": past.target.tail(3).median(),
            "duration_iqr": past.target.quantile(.75) - past.target.quantile(.25),
            "since_last_plugout_hours": (session.origin - past.available.iloc[-1]).total_seconds() / 3600,
            **dict(zip(DONOR_NAMES, donor_values)),
        })
    return base.join(pd.DataFrame(rows).set_index("index"), how="inner")


def benchmark_hybrid(train, test, cutoff):
    assert not set(train.user_id) & set(test.user_id)
    assert train.available.lt(cutoff).all() and test.origin.ge(cutoff).all()

    @lru_cache(maxsize=None)
    def bank_at(timestamp, n):
        available = train.loc[train.available.lt(timestamp)]
        references, users, profiles = {}, [], []
        for user, group in available.groupby("user_id", sort=True):
            if len(group) < n:
                continue
            recent = group.sort_values(["available", "session_id"]).tail(n)
            references[user] = (clock(group.origin), group.target.to_numpy())
            users.append(user)
            profiles.append(profile(recent) / SCALE)
        return references, np.asarray(users, dtype=str), np.asarray(profiles).reshape(-1, len(SCALE))

    train_base, test_base = features(train, train), features(test, test)
    train_x = checkpoint_features(train, train_base, bank_at)
    test_x = checkpoint_features(test, test_base, bank_at, cutoff)
    basic = train_base.columns.drop("last").tolist()
    personal = basic + PROFILE_NAMES + ["last_duration", "recent_median", "duration_iqr",
                                        "since_last_plugout_hours"]

    def fit_predict(columns, training):
        model = build_model()
        location_encoder = model.named_steps["features"].transformers[1]
        model.set_params(
            features__transformers=[
                ("numeric", "passthrough", [column for column in columns if column != "location"]),
                location_encoder,
            ],
            model__loss="absolute_error", model__early_stopping=False,
        )
        return model.fit(training, train.loc[training.index, "target"]).predict(test_x)

    errors = test_x[["user", "history"]].copy()
    actual = test.loc[test_x.index, "target"]
    predictions = {
        "original_boosting": fit_predict(basic, train_base),
        "checkpoint_boosting": fit_predict(basic, train_x),
        "personal_history": fit_predict(personal, train_x),
        "personal_plus_donors": fit_predict(personal + DONOR_NAMES, train_x),
    }
    for method, prediction in predictions.items():
        errors[method] = np.abs(prediction - actual)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for n, group in errors.groupby("history"):
        draws = rng.integers(len(group), size=(5000, len(group)))
        for method in predictions:
            row = dict(history=n, users=len(group), method=method, mae_hours=group[method].mean())
            for baseline in ("original_boosting", "personal_history"):
                gain = (group[baseline] - group[method]).to_numpy()
                low, high = np.quantile(gain[draws].mean(axis=1), [.025, .975])
                row.update({f"gain_vs_{baseline}": gain.mean(),
                            f"{baseline}_ci_low": low, f"{baseline}_ci_high": high})
            rows.append(row)
    metadata = dict(
        cutoff=str(cutoff), reference_sessions=len(train), training_checkpoints=len(train_x),
        training_checkpoints_with_donors=int(train_x.donor_count.gt(0).sum()),
        test_checkpoints=len(test_x), seed=RANDOM_SEED,
        test_users_by_history={str(n): int(count) for n, count in test_x.history.value_counts().items()},
    )
    return pd.DataFrame(rows), metadata


def run(root):
    train, test, cutoff, held_users = pilot_split(root)
    metrics, metadata = benchmark_hybrid(train, test, cutoff)
    out = root / "outputs/tables/similar_users_hybrid"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.6f")
    (out / "metadata.json").write_text(json.dumps(dict(metadata, held_users=held_users), indent=2),
                                      encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for method, label in zip(metrics.method.unique(), ["Original boosting", "Boosting on checkpoints",
                                          "Richer personal history", "Personal history + similar users"]):
        scores = metrics.loc[metrics.method.eq(method)]
        ax.plot(scores.history, scores.mae_hours, "o-", label=label)
    counts = metrics.groupby("history").users.first()
    ax.set(xticks=counts.index, xticklabels=[f"{n}\n({count} users)" for n, count in counts.items()],
           xlabel="Completed sessions available at forecast time", ylabel="Mean absolute error (h)",
           title="Do similar-user features improve connection-duration boosting?")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    folder = root / "outputs/figures/audited_ml"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "04_similar_users_hybrid.png", dpi=180)
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(run(Path(__file__).resolve().parents[2]).to_string(index=False))
