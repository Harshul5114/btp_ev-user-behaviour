"""Audit CrowdCharge transactions and repeat the similar-user benchmark."""

import json
from pathlib import Path

from .similar_users import RANDOM_SEED, benchmark

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_sessions(root):
    source = root / "data/raw/electric_nation/CrowdCharge Transactions.xlsx"
    raw = pd.read_excel(source, sheet_name="Transaction Data", usecols=[
        "TransactionID", "ParticipantID", "ChargerID", "StartTime", "StopTime",
        "AdjustedStartTime", "AdjustedStopTime", "ConsumedkWh", "Trial",
    ], dtype={"ParticipantID": "string", "TransactionID": "string", "ChargerID": "string"})
    origin = pd.to_datetime(raw.StartTime, errors="coerce", utc=True).dt.tz_convert("Europe/London")
    available = pd.to_datetime(raw.StopTime, errors="coerce", utc=True).dt.tz_convert("Europe/London")
    clock_matches = (origin.dt.tz_localize(None).eq(raw.AdjustedStartTime)
                     & available.dt.tz_localize(None).eq(raw.AdjustedStopTime))
    duration = (available - origin).dt.total_seconds() / 3600
    participant = raw.ParticipantID.str.strip()
    missing = participant.isna() | participant.eq("").fillna(False)
    if raw.TransactionID.isna().any() or raw.TransactionID.duplicated().any():
        raise ValueError("Transaction IDs must be present and unique")
    identified = raw.loc[~missing].assign(ParticipantID=participant.loc[~missing])
    counts = {
        "raw_transactions": len(raw),
        "raw_chargers": int(raw.ChargerID.nunique()),
        "identified_participants": int(participant.loc[~missing].nunique()),
        "participants_with_multiple_chargers": int(identified.groupby("ParticipantID").ChargerID.nunique().gt(1).sum()),
        "chargers_with_multiple_participants": int(identified.groupby("ChargerID").ParticipantID.nunique().gt(1).sum()),
        "local_timestamp_disagreements": int((~clock_matches).sum()),
        "trial_counts_raw": raw.Trial.fillna(0).astype(int).value_counts().sort_index().to_dict(),
    }
    # Apply and count filters sequentially so exclusions reconcile with the cohort.
    keep = pd.Series(True, index=raw.index)
    for label, valid in (
        ("missing_participant", ~missing),
        ("missing_or_out_of_range_energy", raw.ConsumedkWh.between(.5, 100)),
        ("invalid_or_under_15_min_duration", duration.ge(.25)),
        ("duration_over_seven_days", duration.le(168)),
    ):
        counts[label] = int((keep & ~valid).sum())
        keep &= valid
    sessions = pd.DataFrame({
        "user_id": participant, "session_id": raw.TransactionID,
        "origin": origin, "available": available, "target": duration,
        "location": "CrowdCharge", "trial": raw.Trial.fillna(0).astype(int),
        "energy_kwh": raw.ConsumedkWh,
    }).loc[keep].sort_values(["origin", "session_id"])
    same_user = sessions.groupby("user_id", sort=False)
    previous_end = same_user.available.transform(lambda holds: holds.cummax().shift())
    next_start = same_user.origin.shift(-1)
    overlapping = sessions.origin.lt(previous_end) | sessions.available.gt(next_start)
    counts["overlap_excluded"] = int(overlapping.sum())
    sessions = sessions.loc[~overlapping]
    counts["usable_transactions"] = len(sessions)
    counts["usable_participants"] = int(sessions.user_id.nunique())
    return sessions, counts


def split_sessions(sessions):
    users = sorted(sessions.user_id.unique())
    cutoff = sessions.origin.iloc[int(.8 * len(sessions))]
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=RANDOM_SEED)
    _, held = next(splitter.split(users, groups=users))
    held_mask = sessions.user_id.isin({users[index] for index in held})
    train = sessions.loc[~held_mask & sessions.available.lt(cutoff)]
    test = sessions.loc[held_mask & sessions.origin.ge(cutoff)]
    assert not set(train.user_id) & set(test.user_id)
    assert train.available.lt(cutoff).all() and test.origin.ge(cutoff).all()
    return train, test, cutoff, len(held)


def run(root):
    sessions, counts = load_sessions(root)
    train, test, cutoff, held_users = split_sessions(sessions)
    metrics = benchmark(train, test)
    metadata = dict(
        counts=counts, cutoff=str(cutoff), train_sessions=len(train), test_sessions=len(test),
        train_users=int(train.user_id.nunique()), test_users=int(test.user_id.nunique()),
        held_users=held_users, seed=RANDOM_SEED, median_duration_hours=float(sessions.target.median()),
        first_session=str(sessions.origin.min()), last_session=str(sessions.origin.max()),
        train_trial_counts=train.trial.value_counts().sort_index().to_dict(),
        test_trial_counts=test.trial.value_counts().sort_index().to_dict(),
    )
    out = root / "outputs/tables/electric_nation"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "similar_users_metrics.csv", index=False, float_format="%.6f")
    (out / "audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
