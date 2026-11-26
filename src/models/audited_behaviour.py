"""Compare charging behaviour predictors using event-time information.

Run with python -m src.models.audited_behaviour from the repository root.
"""

from collections import defaultdict
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .session_energy import (
    RANDOM_SEED, _time_features, build_model, load_reported_sessions,
)


def load_targets(path):
    sessions = load_reported_sessions(path)
    sessions["plugout_dt"] = pd.to_datetime(sessions.plugout_time, errors="coerce")
    sessions["next_plugin_dt"] = sessions.groupby("user_id").plugin_dt.shift(-1)
    duration = (sessions.plugout_dt - sessions.plugin_dt).dt.total_seconds() / 3600
    gap = (sessions.next_plugin_dt - sessions.plugout_dt).dt.total_seconds() / 3600
    completed = duration.gt(0)
    targets = {
        "next_plugin_hours": (completed & gap.ge(0), "plugout_dt", "next_plugin_dt", gap),
        "connection_hours": (completed, "plugin_dt", "plugout_dt", duration),
        "energy_kwh": (completed, "plugin_dt", "plugout_dt", sessions.energy_session),
    }
    counts = {
        "source_sessions": len(sessions),
        "missing_plugout": int(sessions.plugout_dt.isna().sum()),
        "nonpositive_connection": int(duration.le(0).sum()),
        "no_recorded_successor": int(sessions.next_plugin_dt.isna().sum()),
        "negative_gap_excluded": int((completed & gap.lt(0)).sum()),
    }
    frames = {}
    for name, (valid, origin, available, value) in targets.items():
        frame = sessions.loc[valid, ["user_id", "session_id", "location", origin, available]].copy()
        frame = frame.rename(columns={origin: "origin", available: "available"})
        frame["target"] = value[valid]
        frames[name] = frame
    return frames, counts, sessions.plugin_dt.iloc[int(.8 * len(sessions))]


def features(records, observations):
    """A label may enter history strictly after the event that reveals it."""
    clock_records = records.rename(columns={"origin": "plugin_dt"})
    result = _time_features(clock_records)
    events = iter(observations.sort_values(["available", "session_id"]).itertuples())
    pending = next(events, None)
    histories = defaultdict(list)
    rows = []
    for row in records.sort_values(["origin", "session_id"]).itertuples():
        while pending is not None and pending.available < row.origin:
            histories[pending.user_id].append(pending.target)
            pending = next(events, None)
        history = histories[row.user_id]
        rows.append((row.Index, len(history), np.mean(history) if history else 0.,
                     np.median(history) if history else 0., history[-1] if history else np.nan))
    history_frame = pd.DataFrame(rows, columns=["index", "history_count", "history_mean_kwh",
                                               "history_median_kwh", "last"]).set_index("index")
    return result.join(history_frame)


def evaluate(name, protocol, train, test, observed_test):
    train_x = features(train, train)
    test_x = features(test, observed_test)
    median = train.target.median()
    calendar = train.assign(weekday=train.origin.dt.dayofweek, hour=train.origin.dt.hour)
    calendar = calendar.groupby(["weekday", "hour"]).target.median()
    hour_keys = pd.MultiIndex.from_arrays([test.origin.dt.dayofweek, test.origin.dt.hour])
    history = test_x.history_count.gt(0)
    model = build_model().set_params(
        model__loss="absolute_error", model__early_stopping=False,
    )
    model.fit(train_x, train.target)
    predictions = {
        "training_median": median,
        "calendar_median": calendar.reindex(hour_keys).fillna(median).to_numpy(),
        "last_observed": test_x["last"].fillna(median),
        "history_median": test_x.history_median_kwh.where(history, median),
        "gradient_boosting_mae": model.predict(test_x),
    }
    seen = test.user_id.isin(train.user_id)
    masks = {"all": np.ones(len(test), dtype=bool)}
    if protocol == "chronological":
        masks.update(seen_in_training=seen, new_at_cutoff=~seen, zero_history=~history)
    rows = []
    for group, mask in masks.items():
        subset = test.loc[mask]
        if subset.empty:
            continue
        for method, prediction in predictions.items():
            error = np.abs((np.broadcast_to(prediction, len(test)) - test.target.to_numpy())[mask])
            rows.append(dict(target=name, evaluation=protocol, subset=group, predictor=method,
                             sessions=len(subset), users=subset.user_id.nunique(),
                             mae=error.mean(), rmse=np.sqrt(np.mean(error ** 2)),
                             within_6h=float((error <= 6).mean()) if name != "energy_kwh" else None,
                             within_24h=float((error <= 24).mean()) if name != "energy_kwh" else None))
    paired = test[["user_id", "target"]].copy()
    paired["history_count"] = test_x.history_count
    for method in ("training_median", "history_median", "gradient_boosting_mae"):
        paired[method] = np.abs(predictions[method] - test.target)
    return rows, paired


def run(root):
    path = root / "data/raw/norway_ev/Dataset1_charging_reports.csv"
    targets, counts, cutoff = load_targets(path)
    # Use the same held-out users for every target. This is a separate, non-temporal test.
    users = sorted(set().union(*(frame.user_id for frame in targets.values())))
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=RANDOM_SEED)
    _, held = next(splitter.split(users, groups=users))
    held_users = {users[index] for index in held}
    results = []
    chronological = {}
    splits = {}
    for name, frame in targets.items():
        train = frame.loc[frame.available.lt(cutoff) & frame.origin.lt(cutoff)]
        test = frame.loc[frame.origin.ge(cutoff)]
        splits[name] = dict(eligible=len(frame), training=len(train), test=len(test),
                            purged=int((frame.origin.lt(cutoff) & frame.available.ge(cutoff)).sum()))
        rows, chronological[name] = evaluate(name, "chronological", train, test, frame)
        results.extend(rows)
        group_test = frame.user_id.isin(held_users)
        rows, _ = evaluate(name, "unseen_user", frame.loc[~group_test],
                           frame.loc[group_test], frame.loc[~group_test])
        results.extend(rows)
        print(f"{name}: {splits[name]}", flush=True)
    out = root / "outputs/tables/audited_ml"
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(results)
    metrics.to_csv(out / "metrics.csv", index=False, float_format="%.6f")
    metadata = dict(counts=counts, splits=splits, cutoff=str(cutoff), random_seed=RANDOM_SEED,
                    held_out_users=len(held_users))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    from .behaviour_diagnostics import save_diagnostics
    save_diagnostics(root, chronological, RANDOM_SEED)
    return metrics


if __name__ == "__main__":
    results = run(Path(__file__).resolve().parents[2])
    print(results.loc[results.subset.eq("all")].to_string(index=False))
