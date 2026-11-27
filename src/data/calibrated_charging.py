"""Generate connection sessions using empirical training-cohort distributions."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.models.audited_behaviour import load_reported_sessions
from src.models.electric_nation import load_sessions, split_sessions
from src.models.similar_users import pilot_split, profile


# Pool weights in order: similar real drivers, one real driver, entire source.
SCENARIOS = {
    "shared_routines": (.85, .10, .05),
    "individual_routines": (.10, .85, .05),
    "irregular_routines": (.10, .10, .80),
}
TRAIN_START = pd.Timestamp("2018-01-01")
CUTOFF = pd.Timestamp("2020-01-01")


def calibration(root):
    norway, _, norway_cutoff, _ = pilot_split(root)
    norway_source = root / "data/raw/norway_ev/Dataset1_charging_reports.csv"
    energy = load_reported_sessions(norway_source).set_index("session_id").energy_session
    norway = norway.assign(energy_kwh=norway.session_id.map(energy))
    uk_data, _ = load_sessions(root)
    uk, _, uk_cutoff, _ = split_sessions(uk_data)
    references, metadata = {}, {}
    for source, rows, cutoff in (("Norway", norway, norway_cutoff), ("Electric Nation", uk, uk_cutoff)):
        # Apply common calibration bounds and use only driver histories with support.
        selected = rows.target.between(.25, 72) & rows.energy_kwh.between(.5, 100)
        filtered = rows.loc[selected].sort_values(["origin", "session_id"])
        assert filtered.available.lt(cutoff).all()
        donors, summaries = {}, []
        for user, group in filtered.groupby("user_id", sort=True):
            if len(group) < 15:
                continue
            # Clock values are local. Gaps in calendar days are paired with
            # the session that follows each gap, then capped for this study.
            local = group.origin.dt.tz_localize(None)
            days = local.dt.normalize().diff().dt.days.iloc[1:].clip(0, 10)
            clock_min = local.dt.hour * 60 + local.dt.minute
            donors[user] = np.column_stack([
                clock_min.iloc[1:], group.target.iloc[1:] * 60,
                group.energy_kwh.iloc[1:], days, local.dt.dayofweek.iloc[1:].ge(5),
            ])
            summaries.append(np.r_[profile(group), np.log1p(days.median())])
        ids = list(donors)
        standardized = StandardScaler().fit_transform(summaries)
        labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(standardized)
        cluster = dict(zip(ids, labels))
        pools = {int(label): np.concatenate([donors[user] for user in ids if cluster[user] == label])
                 for label in np.unique(labels)}
        population = np.concatenate(list(donors.values()))
        references[source] = (ids, donors, cluster, pools, population)
        metadata[source] = dict(cutoff=str(cutoff), training_sessions=len(rows),
                                donor_users=len(ids), calibration_rows=len(population),
                                median_duration_hours=float(np.median(population[:, 1]) / 60),
                                median_day_gap=float(np.median(population[:, 3])),
                                cluster_user_counts={str(int(k)): int((labels == k).sum())
                                                     for k in np.unique(labels)})
    return references, metadata


def generate(references, scenario, seed, users=1000, sessions_per_user=48):
    rng = np.random.default_rng(seed)
    rows = []
    # Assign the source and cohort independently of simulated routines.
    order = rng.permutation(users)
    held = set(order[:users // 5])
    sources = list(references)
    weights = SCENARIOS[scenario]
    hour_jitter = 30 if scenario != "irregular_routines" else 75
    day_type_pools = {}
    for index in range(users):
        source = sources[index % len(sources)]
        ids, donors, cluster, pools, population = references[source]
        donor = ids[rng.integers(len(ids))]
        local_pools = (pools[cluster[donor]], donors[donor], population)
        cohort = "test" if index in held else "train"
        start = CUTOFF if cohort == "test" else TRAIN_START
        day = int(rng.integers(0, 45))
        previous_end = -1
        static_jitter = rng.normal(0, 20)
        for number in range(sessions_per_user):
            pool = local_pools[rng.choice(3, p=weights)]
            minute, duration, energy, days, sampled_weekend = pool[rng.integers(len(pool))]
            if number:
                day += int(days)
            # Match the weekday/weekend clock and energy pattern when possible.
            weekend = (start.dayofweek + day) % 7 >= 5
            if bool(sampled_weekend) != weekend:
                key = (id(pool), weekend)
                if key not in day_type_pools:
                    day_type_pools[key] = pool[pool[:, 4] == weekend]
                matches = day_type_pools[key]
                if len(matches):
                    minute, duration, energy = matches[rng.integers(len(matches)), :3]
            minute = int(round(minute + static_jitter + rng.normal(0, hour_jitter))) % 1440
            duration = max(15, int(round(duration * rng.lognormal(0, .10))))
            beginning = day * 1440 + minute
            # Keep sessions disjoint while preserving the sampled plug-in hour.
            if beginning < previous_end + 5:
                days_to_add = (previous_end + 5 - beginning + 1439) // 1440
                day += days_to_add
                beginning += 1440 * days_to_add
            previous_end = beginning + duration
            rows.append((f"S{index:05d}-{number:03d}", f"U{index:05d}", source, cohort,
                         start + pd.Timedelta(minutes=beginning),
                         start + pd.Timedelta(minutes=previous_end), duration / 60, energy))
    return pd.DataFrame(rows, columns=["session_id", "user_id", "location", "cohort",
                                       "origin", "available", "target", "energy_kwh"])
