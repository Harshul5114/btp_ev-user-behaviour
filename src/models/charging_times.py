"""Leakage-free ML pipeline for EV plug-in and plug-out time prediction.

Formulation:
1. Plug-out Time:
   - Predicted at `plugin_time`.
   - Target: `connection_time` (duration in hours from `plugin_time` to `plugout_time`).
   - Formulation: Overnight-aware setup incorporating out-of-fold Stage-1
     classifier probability P(overnight) alongside quantile regressors
     (p10, p50, p90) and point estimates (L1 / absolute error loss).
   - Recovered plug-out timestamp: plugin_time + connection_time.

2. Next Plug-in Time:
   - Predicted at current session's `plugout_time`.
   - Target: `next_gap_hours` (duration in hours from current `plugout_time` to
     the user's next `plugin_time`).
   - Quantile regressors (p10, p50, p90) and point estimates (L1 loss).
   - Recovered next plug-in timestamp: current_plugout_time + next_gap_hours.
   - Circular clock-hour evaluation on recovered plug-in hour.

Causal Guarantees:
- Calendar & day-of-week context strictly at prediction time.
- NHTS empirical mobility priors (Home arrival probability, first Home departure
  probability, expected Home stay duration) indexed by (hour, is_weekend).
- Causal user history: expanding & rolling statistics (last 3, 5, 10 sessions,
  shifted by 1), separate weekday/weekend user medians, circular sin/cos mean of
  prior plug-in/plug-out hours, hours since last plugout, and estimated energy deficit.
- Site context: smoothed out-of-fold target encoding for location fit strictly on
  each training split.
- Exact evaluation splits: chronological 80/20 future-session split (all_test,
  warm_user, cold_start_user) and 20% unseen-user GroupShuffleSplit holdout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Avoid Matplotlib non-writable config dir warnings in restricted execution environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, KFold


RANDOM_SEED = 42
TARGET_CONNECTION = "connection_time"
TARGET_GAP = "next_gap_hours"


@dataclass
class NHTSPriors:
    """Empirical prior distributions derived from NHTS trip chains and parking data."""

    p_home_arrival: np.ndarray  # Shape (2, 24): [is_weekend, hour]
    p_first_home_departure: np.ndarray  # Shape (2, 24)
    expected_home_stay_hours: np.ndarray  # Shape (2, 24)
    overall_mean_stay: float

    def lookup(self, is_weekend: int, hour: int) -> Tuple[float, float, float]:
        """Return (p_home_arr, p_first_dept, expected_stay) for (is_weekend, hour)."""
        we = int(np.clip(is_weekend, 0, 1))
        h = int(hour) % 24
        return (
            float(self.p_home_arrival[we, h]),
            float(self.p_first_home_departure[we, h]),
            float(self.expected_home_stay_hours[we, h]),
        )


@dataclass
class UserHistoryState:
    """Tracks causal historical observations for a single user."""

    user_id: str
    conns: List[float] = field(default_factory=list)
    conns_wd: List[float] = field(default_factory=list)
    conns_we: List[float] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    energies_wd: List[float] = field(default_factory=list)
    energies_we: List[float] = field(default_factory=list)
    gaps: List[float] = field(default_factory=list)
    gaps_wd: List[float] = field(default_factory=list)
    gaps_we: List[float] = field(default_factory=list)
    in_sin: List[float] = field(default_factory=list)
    in_cos: List[float] = field(default_factory=list)
    out_sin: List[float] = field(default_factory=list)
    out_cos: List[float] = field(default_factory=list)
    last_plugout_dt: Optional[pd.Timestamp] = None
    first_plugin_dt: Optional[pd.Timestamp] = None


def load_clean_charging_data(raw_path: Path) -> pd.DataFrame:
    """Load and validate Norway Dataset 1 charging reports.

    Ensures valid timestamps and nonnegative values for connection_time and energy_session.
    Sorts chronologically by plugin_dt and session_id.
    """
    sessions = pd.read_csv(
        raw_path,
        sep=";",
        decimal=",",
        dtype={"location": "string", "user_id": "string", "session_id": "string"},
        low_memory=False,
    )
    sessions["plugin_dt"] = pd.to_datetime(sessions["plugin_time"], errors="coerce")
    sessions["plugout_dt"] = pd.to_datetime(sessions["plugout_time"], errors="coerce")
    sessions["connection_time"] = pd.to_numeric(sessions["connection_time"], errors="coerce")
    sessions["energy_session"] = pd.to_numeric(sessions["energy_session"], errors="coerce")

    valid = sessions.loc[
        sessions["plugin_dt"].notna()
        & sessions["plugout_dt"].notna()
        & sessions["connection_time"].notna()
        & sessions["connection_time"].gt(0)
        & sessions["energy_session"].notna()
        & sessions["energy_session"].ge(0)
    ].copy()

    # Sort strictly chronologically by plugin time, then session_id for deterministic order
    valid = valid.sort_values(["plugin_dt", "session_id"], kind="stable").reset_index(drop=True)

    # Calculate user-specific next session plugin time and next gap hours
    valid = valid.sort_values(["user_id", "plugin_dt"], kind="stable").reset_index(drop=True)
    valid["next_plugin_dt"] = valid.groupby("user_id")["plugin_dt"].shift(-1)
    # Gap in hours between current plug-out and next plug-in
    valid["next_gap_hours"] = (
        (valid["next_plugin_dt"] - valid["plugout_dt"]).dt.total_seconds() / 3600.0
    )
    # Small negative gaps (<0.2% of data) due to logger overlap clipped cleanly to 0
    valid["next_gap_hours"] = valid["next_gap_hours"].clip(lower=0.0)

    # Re-sort strictly by plugin_dt across all users
    return valid.sort_values(["plugin_dt", "session_id"], kind="stable").reset_index(drop=True)


def build_nhts_priors(chains_path: Path, park_path: Path) -> NHTSPriors:
    """Build empirical prior lookup tables from clean NHTS vehicle trip chains and parking data."""
    chains = pd.read_csv(chains_path, low_memory=False)
    park = pd.read_csv(park_path, low_memory=False)

    chains["is_weekend"] = chains["TRAVDAY"].isin([1, 7]).astype(int)
    chains["arr_hour"] = (chains["end_min"] // 60).astype(int) % 24
    chains["dept_hour"] = (chains["start_min"] // 60).astype(int) % 24

    # 1. Weighted Home arrival probability density across 24 hours
    home_trips = chains[chains["destination_group"] == "Home"].copy()
    home_arr_wt = home_trips.groupby(["is_weekend", "arr_hour"])["WTTRDFIN"].sum()
    total_home_wt = home_trips.groupby("is_weekend")["WTTRDFIN"].sum()
    p_arr_df = (
        (home_arr_wt / total_home_wt)
        .unstack(fill_value=0.0)
        .reindex(index=[0, 1], columns=range(24), fill_value=0.0)
    )

    # 2. Weighted first Home departure probability density across 24 hours
    first_home = chains[(chains["vehicle_leg_index"] == 1) & (chains["begins_home"])].copy()
    first_dept_wt = first_home.groupby(["is_weekend", "dept_hour"])["WTTRDFIN"].sum()
    total_first_wt = first_home.groupby("is_weekend")["WTTRDFIN"].sum()
    p_dept_df = (
        (first_dept_wt / total_first_wt)
        .unstack(fill_value=0.0)
        .reindex(index=[0, 1], columns=range(24), fill_value=0.0)
    )

    # 3. Expected Home stay duration across 24 hours
    merged_park = park.merge(
        chains[["VEHCASEID", "TDCASEID", "is_weekend", "WTTRDFIN"]].drop_duplicates(),
        on=["VEHCASEID", "TDCASEID"],
        how="left",
    )
    home_park = merged_park[
        (merged_park["destination_group"] == "Home") & (merged_park["analysis_eligible"])
    ].copy()
    home_park["park_hour"] = (home_park["parking_start_min"] // 60).astype(int) % 24
    home_park["stay_hours"] = home_park["parking_duration_min"] / 60.0

    def w_mean(g: pd.DataFrame) -> float:
        w = g["WTTRDFIN"].fillna(1.0)
        return float(np.average(g["stay_hours"], weights=w)) if len(g) > 0 else np.nan

    stay_grouped = home_park.groupby(["is_weekend", "park_hour"]).apply(w_mean, include_groups=False).unstack()
    overall_stay = float(
        np.average(home_park["stay_hours"], weights=home_park["WTTRDFIN"].fillna(1.0))
    )
    stay_df = (
        stay_grouped.reindex(index=[0, 1], columns=range(24))
        .interpolate(axis=1, limit_direction="both")
        .fillna(overall_stay)
    )

    return NHTSPriors(
        p_home_arrival=p_arr_df.to_numpy(),
        p_first_home_departure=p_dept_df.to_numpy(),
        expected_home_stay_hours=stay_df.to_numpy(),
        overall_mean_stay=overall_stay,
    )


def _get_norway_holidays() -> Set:
    """Precompute set of Norwegian public holiday dates for 2018-2022."""
    no_holidays = holidays.country_holidays("NO", years=range(2018, 2023))
    return set(no_holidays.keys())


def smoothed_target_encoding(
    train_locs: pd.Series,
    train_y: pd.Series,
    test_locs: pd.Series,
    m: float = 10.0,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothed out-of-fold target encoding for location fit strictly on training data."""
    global_mean = float(train_y.mean())
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    train_encoded = pd.Series(index=train_locs.index, dtype=float)
    for tr_idx, val_idx in kf.split(train_locs):
        tr_l, tr_target = train_locs.iloc[tr_idx], train_y.iloc[tr_idx]
        stats = tr_target.groupby(tr_l).agg(["count", "mean"])
        val_l = train_locs.iloc[val_idx]
        val_counts = val_l.map(stats["count"]).fillna(0.0)
        val_means = val_l.map(stats["mean"]).fillna(global_mean)
        train_encoded.iloc[val_idx] = (val_counts * val_means + m * global_mean) / (
            val_counts + m
        )

    # For test data: use complete training statistics
    full_stats = train_y.groupby(train_locs).agg(["count", "mean"])
    test_counts = test_locs.map(full_stats["count"]).fillna(0.0)
    test_means = test_locs.map(full_stats["mean"]).fillna(global_mean)
    test_encoded = (test_counts * test_means + m * global_mean) / (test_counts + m)

    return train_encoded.to_numpy(), test_encoded.to_numpy()


def extract_connection_time_features(
    records: pd.DataFrame,
    priors: NHTSPriors,
    initial_histories: Optional[Dict[str, UserHistoryState]] = None,
    update_history: bool = True,
    fallback_stats: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, UserHistoryState], Dict[str, float]]:
    """Extract causal features available at plug-in time to predict connection_time.

    All history features strictly use completed prior sessions of that user.
    """
    ordered = records.sort_values(["plugin_dt", "session_id"], kind="stable").copy()
    histories = {
        u: UserHistoryState(
            user_id=u,
            conns=list(h.conns),
            conns_wd=list(h.conns_wd),
            conns_we=list(h.conns_we),
            energies=list(h.energies),
            energies_wd=list(h.energies_wd),
            energies_we=list(h.energies_we),
            gaps=list(h.gaps),
            gaps_wd=list(h.gaps_wd),
            gaps_we=list(h.gaps_we),
            in_sin=list(h.in_sin),
            in_cos=list(h.in_cos),
            out_sin=list(h.out_sin),
            out_cos=list(h.out_cos),
            last_plugout_dt=h.last_plugout_dt,
            first_plugin_dt=h.first_plugin_dt,
        )
        for u, h in (initial_histories or {}).items()
    }

    if fallback_stats is None:
        fb_conn = float(ordered["connection_time"].median())
        fb_energy = float(ordered["energy_session"].median())
        fb_gap = 10.0
        fallback_stats = {
            "fb_conn": fb_conn,
            "fb_energy": fb_energy,
            "fb_gap": fb_gap,
        }
    else:
        fb_conn = fallback_stats["fb_conn"]
        fb_energy = fallback_stats["fb_energy"]
        fb_gap = fallback_stats["fb_gap"]

    no_holidays = _get_norway_holidays()
    feature_rows: List[Dict[str, float]] = []

    for row in ordered.itertuples(index=False):
        u = str(row.user_id)
        if u not in histories:
            histories[u] = UserHistoryState(user_id=u)
        uh = histories[u]

        # 1. Calendar & Day-of-week context at plugin_time
        dt = row.plugin_dt
        hour_clock = dt.hour + dt.minute / 60.0
        hour_int = dt.hour
        dow = dt.dayofweek
        is_we = int(dow >= 5)
        doy = dt.dayofyear
        is_holiday = int(dt.date() in no_holidays)

        next_date = (dt + pd.Timedelta(days=1)).date()
        next_dow = (dow + 1) % 7
        next_day_is_workday = int((next_dow < 5) and (next_date not in no_holidays))

        # 2. NHTS Empirical Priors
        p_home_arr, p_first_dept, exp_stay = priors.lookup(is_we, hour_int)

        # 3. Causal User History & Usage Urgency
        k = len(uh.conns)
        has_prior = 1.0 if k > 0 else 0.0

        if uh.last_plugout_dt is not None:
            hours_since_plugout = max(
                0.0, (dt - uh.last_plugout_dt).total_seconds() / 3600.0
            )
        else:
            hours_since_plugout = fb_gap

        # Medians and IQRs
        if k > 0:
            hist_med_conn = float(np.median(uh.conns))
            hist_iqr_conn = float(
                np.percentile(uh.conns, 75) - np.percentile(uh.conns, 25)
            ) if k >= 4 else 0.0
            hist_med_energy = float(np.median(uh.energies))
            hist_med_gap = float(np.median(uh.gaps)) if uh.gaps else fb_gap

            hist_wd_med_conn = float(np.median(uh.conns_wd)) if uh.conns_wd else hist_med_conn
            hist_we_med_conn = float(np.median(uh.conns_we)) if uh.conns_we else hist_med_conn

            hist_wd_med_energy = float(np.median(uh.energies_wd)) if uh.energies_wd else hist_med_energy
            hist_we_med_energy = float(np.median(uh.energies_we)) if uh.energies_we else hist_med_energy

            hist_wd_med_gap = float(np.median(uh.gaps_wd)) if uh.gaps_wd else hist_med_gap
            hist_we_med_gap = float(np.median(uh.gaps_we)) if uh.gaps_we else hist_med_gap

            roll3_med_conn = float(np.median(uh.conns[-3:]))
            roll3_iqr_conn = float(
                np.percentile(uh.conns[-3:], 75) - np.percentile(uh.conns[-3:], 25)
            ) if k >= 3 else 0.0

            roll5_med_conn = float(np.median(uh.conns[-5:]))
            roll5_iqr_conn = float(
                np.percentile(uh.conns[-5:], 75) - np.percentile(uh.conns[-5:], 25)
            ) if k >= 4 else 0.0

            roll10_med_conn = float(np.median(uh.conns[-10:]))
            roll10_iqr_conn = float(
                np.percentile(uh.conns[-10:], 75) - np.percentile(uh.conns[-10:], 25)
            ) if k >= 4 else 0.0

            roll5_med_energy = float(np.median(uh.energies[-5:]))

            prior_in_sin = float(np.mean(uh.in_sin))
            prior_in_cos = float(np.mean(uh.in_cos))
            prior_out_sin = float(np.mean(uh.out_sin))
            prior_out_cos = float(np.mean(uh.out_cos))

            # Daily burn rate & deficit
            if uh.first_plugin_dt is not None and uh.last_plugout_dt is not None:
                days_span = max(
                    1.0, (uh.last_plugout_dt - uh.first_plugin_dt).total_seconds() / 86400.0
                )
                daily_burn_rate = float(sum(uh.energies) / days_span)
            else:
                daily_burn_rate = hist_med_energy
            est_deficit = float(daily_burn_rate * hours_since_plugout / 24.0)
        else:
            hist_med_conn = fb_conn
            hist_iqr_conn = 0.0
            hist_med_energy = fb_energy
            hist_med_gap = fb_gap
            hist_wd_med_conn = fb_conn
            hist_we_med_conn = fb_conn
            hist_wd_med_energy = fb_energy
            hist_we_med_energy = fb_energy
            hist_wd_med_gap = fb_gap
            hist_we_med_gap = fb_gap
            roll3_med_conn = fb_conn
            roll3_iqr_conn = 0.0
            roll5_med_conn = fb_conn
            roll5_iqr_conn = 0.0
            roll10_med_conn = fb_conn
            roll10_iqr_conn = 0.0
            roll5_med_energy = fb_energy
            prior_in_sin = 0.0
            prior_in_cos = 0.0
            prior_out_sin = 0.0
            prior_out_cos = 0.0
            daily_burn_rate = fb_energy
            est_deficit = float(daily_burn_rate * hours_since_plugout / 24.0)

        # Baseline point prediction for causal user history
        causal_user_baseline = hist_we_med_conn if is_we else hist_wd_med_conn

        feature_rows.append(
            {
                "plugin_hour_sin": np.sin(2 * np.pi * hour_clock / 24.0),
                "plugin_hour_cos": np.cos(2 * np.pi * hour_clock / 24.0),
                "plugin_weekday_sin": np.sin(2 * np.pi * dow / 7.0),
                "plugin_weekday_cos": np.cos(2 * np.pi * dow / 7.0),
                "plugin_weekend": float(is_we),
                "next_day_is_workday": float(next_day_is_workday),
                "plugin_season_sin": np.sin(2 * np.pi * doy / 365.25),
                "plugin_season_cos": np.cos(2 * np.pi * doy / 365.25),
                "is_holiday": float(is_holiday),
                "nhts_p_home_arrival": p_home_arr,
                "nhts_p_first_home_departure": p_first_dept,
                "nhts_expected_home_stay_hours": exp_stay,
                "history_count": float(k),
                "has_prior_session": has_prior,
                "hours_since_last_plugout": hours_since_plugout,
                "user_hist_median_conn": hist_med_conn,
                "user_hist_iqr_conn": hist_iqr_conn,
                "user_hist_weekday_median_conn": hist_wd_med_conn,
                "user_hist_weekend_median_conn": hist_we_med_conn,
                "user_hist_median_energy": hist_med_energy,
                "user_hist_weekday_median_energy": hist_wd_med_energy,
                "user_hist_weekend_median_energy": hist_we_med_energy,
                "user_hist_median_gap": hist_med_gap,
                "user_hist_weekday_median_gap": hist_wd_med_gap,
                "user_hist_weekend_median_gap": hist_we_med_gap,
                "user_roll3_median_conn": roll3_med_conn,
                "user_roll3_iqr_conn": roll3_iqr_conn,
                "user_roll5_median_conn": roll5_med_conn,
                "user_roll5_iqr_conn": roll5_iqr_conn,
                "user_roll10_median_conn": roll10_med_conn,
                "user_roll10_iqr_conn": roll10_iqr_conn,
                "user_roll5_median_energy": roll5_med_energy,
                "prior_plugin_hour_mean_sin": prior_in_sin,
                "prior_plugin_hour_mean_cos": prior_in_cos,
                "prior_plugout_hour_mean_sin": prior_out_sin,
                "prior_plugout_hour_mean_cos": prior_out_cos,
                "prior_daily_kwh_burn_rate": daily_burn_rate,
                "estimated_energy_deficit": est_deficit,
                "causal_user_baseline": causal_user_baseline,
            }
        )

        if update_history:
            c = float(row.connection_time)
            e = float(row.energy_session)
            h_out = row.plugout_dt.hour + row.plugout_dt.minute / 60.0

            uh.conns.append(c)
            if is_we:
                uh.conns_we.append(c)
            else:
                uh.conns_wd.append(c)
            uh.energies.append(e)
            if is_we:
                uh.energies_we.append(e)
            else:
                uh.energies_wd.append(e)
            if uh.last_plugout_dt is not None:
                uh.gaps.append(hours_since_plugout)
                if is_we:
                    uh.gaps_we.append(hours_since_plugout)
                else:
                    uh.gaps_wd.append(hours_since_plugout)

            uh.in_sin.append(np.sin(2 * np.pi * hour_clock / 24.0))
            uh.in_cos.append(np.cos(2 * np.pi * hour_clock / 24.0))
            uh.out_sin.append(np.sin(2 * np.pi * h_out / 24.0))
            uh.out_cos.append(np.cos(2 * np.pi * h_out / 24.0))

            if uh.first_plugin_dt is None:
                uh.first_plugin_dt = row.plugin_dt
            uh.last_plugout_dt = row.plugout_dt

    features_df = pd.DataFrame(feature_rows, index=ordered.index)
    return features_df, histories, fallback_stats


def extract_next_gap_features(
    records: pd.DataFrame,
    priors: NHTSPriors,
    initial_histories: Optional[Dict[str, UserHistoryState]] = None,
    update_history: bool = True,
    fallback_stats: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, UserHistoryState], Dict[str, float]]:
    """Extract causal features available at current session plug-out time to predict next_gap_hours.

    At current session's plug-out time, the current session has completed.
    Current session's connection_time, energy_session, and plugin_time are thus known.
    """
    ordered = records.sort_values(["plugin_dt", "session_id"], kind="stable").copy()
    histories = {
        u: UserHistoryState(
            user_id=u,
            conns=list(h.conns),
            conns_wd=list(h.conns_wd),
            conns_we=list(h.conns_we),
            energies=list(h.energies),
            energies_wd=list(h.energies_wd),
            energies_we=list(h.energies_we),
            gaps=list(h.gaps),
            gaps_wd=list(h.gaps_wd),
            gaps_we=list(h.gaps_we),
            in_sin=list(h.in_sin),
            in_cos=list(h.in_cos),
            out_sin=list(h.out_sin),
            out_cos=list(h.out_cos),
            last_plugout_dt=h.last_plugout_dt,
            first_plugin_dt=h.first_plugin_dt,
        )
        for u, h in (initial_histories or {}).items()
    }

    if fallback_stats is None:
        fb_gap = float(ordered["next_gap_hours"].dropna().median())
        fb_conn = float(ordered["connection_time"].median())
        fb_energy = float(ordered["energy_session"].median())
        fallback_stats = {
            "fb_gap": fb_gap,
            "fb_conn": fb_conn,
            "fb_energy": fb_energy,
        }
    else:
        fb_gap = fallback_stats["fb_gap"]
        fb_conn = fallback_stats["fb_conn"]
        fb_energy = fallback_stats["fb_energy"]

    no_holidays = _get_norway_holidays()
    feature_rows: List[Dict[str, float]] = []

    for row in ordered.itertuples(index=False):
        u = str(row.user_id)
        if u not in histories:
            histories[u] = UserHistoryState(user_id=u)
        uh = histories[u]

        # Current session completed realizations
        curr_conn = float(row.connection_time)
        curr_energy = float(row.energy_session)
        in_clock = row.plugin_dt.hour + row.plugin_dt.minute / 60.0

        # Plug-out time context
        out_dt = row.plugout_dt
        out_clock = out_dt.hour + out_dt.minute / 60.0
        out_int = out_dt.hour
        out_dow = out_dt.dayofweek
        out_we = int(out_dow >= 5)
        out_doy = out_dt.dayofyear
        out_is_holiday = int(out_dt.date() in no_holidays)

        next_date = (out_dt + pd.Timedelta(days=1)).date()
        next_dow = (out_dow + 1) % 7
        next_day_is_workday = int((next_dow < 5) and (next_date not in no_holidays))

        # NHTS priors at plug-out time
        p_home_arr, p_first_dept, exp_stay = priors.lookup(out_we, out_int)

        # Causal user history (including prior completed gaps up to this plug-out)
        k_gaps = len(uh.gaps)
        has_prior_gaps = 1.0 if k_gaps > 0 else 0.0

        if k_gaps > 0:
            hist_med_gap = float(np.median(uh.gaps))
            hist_wd_med_gap = float(np.median(uh.gaps_wd)) if uh.gaps_wd else hist_med_gap
            hist_we_med_gap = float(np.median(uh.gaps_we)) if uh.gaps_we else hist_med_gap

            roll3_med_gap = float(np.median(uh.gaps[-3:]))
            roll3_iqr_gap = float(
                np.percentile(uh.gaps[-3:], 75) - np.percentile(uh.gaps[-3:], 25)
            ) if k_gaps >= 3 else 0.0

            roll5_med_gap = float(np.median(uh.gaps[-5:]))
            roll5_iqr_gap = float(
                np.percentile(uh.gaps[-5:], 75) - np.percentile(uh.gaps[-5:], 25)
            ) if k_gaps >= 4 else 0.0

            roll10_med_gap = float(np.median(uh.gaps[-10:]))
            roll10_iqr_gap = float(
                np.percentile(uh.gaps[-10:], 75) - np.percentile(uh.gaps[-10:], 25)
            ) if k_gaps >= 4 else 0.0
        else:
            hist_med_gap = fb_gap
            hist_wd_med_gap = fb_gap
            hist_we_med_gap = fb_gap
            roll3_med_gap = fb_gap
            roll3_iqr_gap = 0.0
            roll5_med_gap = fb_gap
            roll5_iqr_gap = 0.0
            roll10_med_gap = fb_gap
            roll10_iqr_gap = 0.0

        k_conns = len(uh.conns)
        hist_med_conn = float(np.median(uh.conns)) if k_conns > 0 else curr_conn
        hist_med_energy = float(np.median(uh.energies)) if k_conns > 0 else curr_energy
        prior_in_sin = float(np.mean(uh.in_sin)) if k_conns > 0 else np.sin(2 * np.pi * in_clock / 24.0)
        prior_in_cos = float(np.mean(uh.in_cos)) if k_conns > 0 else np.cos(2 * np.pi * in_clock / 24.0)
        prior_out_sin = float(np.mean(uh.out_sin)) if k_conns > 0 else np.sin(2 * np.pi * out_clock / 24.0)
        prior_out_cos = float(np.mean(uh.out_cos)) if k_conns > 0 else np.cos(2 * np.pi * out_clock / 24.0)

        causal_user_gap_baseline = hist_we_med_gap if out_we else hist_wd_med_gap

        feature_rows.append(
            {
                "plugout_hour_sin": np.sin(2 * np.pi * out_clock / 24.0),
                "plugout_hour_cos": np.cos(2 * np.pi * out_clock / 24.0),
                "plugout_weekday_sin": np.sin(2 * np.pi * out_dow / 7.0),
                "plugout_weekday_cos": np.cos(2 * np.pi * out_dow / 7.0),
                "plugout_weekend": float(out_we),
                "next_day_is_workday": float(next_day_is_workday),
                "plugout_season_sin": np.sin(2 * np.pi * out_doy / 365.25),
                "plugout_season_cos": np.cos(2 * np.pi * out_doy / 365.25),
                "is_holiday": float(out_is_holiday),
                "curr_connection_time": curr_conn,
                "curr_energy_session": curr_energy,
                "curr_plugin_hour_sin": np.sin(2 * np.pi * in_clock / 24.0),
                "curr_plugin_hour_cos": np.cos(2 * np.pi * in_clock / 24.0),
                "nhts_p_home_arrival": p_home_arr,
                "nhts_p_first_home_departure": p_first_dept,
                "nhts_expected_home_stay_hours": exp_stay,
                "history_count": float(k_conns),
                "has_prior_gaps": has_prior_gaps,
                "user_hist_median_gap": hist_med_gap,
                "user_hist_weekday_median_gap": hist_wd_med_gap,
                "user_hist_weekend_median_gap": hist_we_med_gap,
                "user_roll3_median_gap": roll3_med_gap,
                "user_roll3_iqr_gap": roll3_iqr_gap,
                "user_roll5_median_gap": roll5_med_gap,
                "user_roll5_iqr_gap": roll5_iqr_gap,
                "user_roll10_median_gap": roll10_med_gap,
                "user_roll10_iqr_gap": roll10_iqr_gap,
                "user_hist_median_conn": hist_med_conn,
                "user_hist_median_energy": hist_med_energy,
                "prior_plugin_hour_mean_sin": prior_in_sin,
                "prior_plugin_hour_mean_cos": prior_in_cos,
                "prior_plugout_hour_mean_sin": prior_out_sin,
                "prior_plugout_hour_mean_cos": prior_out_cos,
                "causal_user_gap_baseline": causal_user_gap_baseline,
            }
        )

        if update_history:
            # Register current session into user history
            uh.conns.append(curr_conn)
            if out_we:
                uh.conns_we.append(curr_conn)
            else:
                uh.conns_wd.append(curr_conn)
            uh.energies.append(curr_energy)
            if out_we:
                uh.energies_we.append(curr_energy)
            else:
                uh.energies_wd.append(curr_energy)

            # If this session had an observed next_gap_hours, record it as a past gap for subsequent sessions
            realized_gap = row.next_gap_hours
            if pd.notna(realized_gap):
                g_val = float(realized_gap)
                uh.gaps.append(g_val)
                if out_we:
                    uh.gaps_we.append(g_val)
                else:
                    uh.gaps_wd.append(g_val)

            uh.in_sin.append(np.sin(2 * np.pi * in_clock / 24.0))
            uh.in_cos.append(np.cos(2 * np.pi * in_clock / 24.0))
            uh.out_sin.append(np.sin(2 * np.pi * out_clock / 24.0))
            uh.out_cos.append(np.cos(2 * np.pi * out_clock / 24.0))
            uh.last_plugout_dt = row.plugout_dt

    features_df = pd.DataFrame(feature_rows, index=ordered.index)
    return features_df, histories, fallback_stats


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Calculate average pinball loss for quantile alpha."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def _interval_metrics(
    y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray
) -> Tuple[float, float]:
    """Calculate empirical 80% coverage and average interval width."""
    coverage = float(np.mean((y_true >= p10) & (y_true <= p90)))
    width = float(np.mean(p90 - p10))
    return coverage, width


def _circular_hour_mae(true_hours: np.ndarray, pred_hours: np.ndarray) -> float:
    """Calculate circular mean absolute error on clock hours [0, 24)."""
    diff = np.abs(true_hours - pred_hours)
    circ = np.minimum(diff, 24.0 - diff)
    return float(np.mean(circ))


@dataclass
class TargetEvaluation:
    """Encapsulates datasets, models, predictions, and evaluation metrics for one target."""

    target_name: str
    evaluation_name: str
    train_x: pd.DataFrame
    train_y: pd.Series
    test_x: pd.DataFrame
    test_y: pd.Series
    test_records: pd.DataFrame
    test_users: pd.Series
    warm_mask: np.ndarray
    point_pred: np.ndarray
    p10_pred: np.ndarray
    p50_pred: np.ndarray
    p90_pred: np.ndarray
    median_baseline: np.ndarray
    history_baseline: np.ndarray
    baseline_p10: np.ndarray
    baseline_p90: np.ndarray


def fit_and_evaluate_connection_time(
    train_records: pd.DataFrame,
    test_records: pd.DataFrame,
    priors: NHTSPriors,
    evaluation_name: str,
    allow_test_history: bool,
) -> TargetEvaluation:
    """Fit overnight-aware point and quantile regressors for connection_time."""
    # 1. Feature extraction
    train_feats, train_histories, fb_stats = extract_connection_time_features(
        train_records, priors, update_history=True
    )
    test_feats, _, _ = extract_connection_time_features(
        test_records,
        priors,
        initial_histories=train_histories,
        update_history=allow_test_history,
        fallback_stats=fb_stats,
    )

    # 2. Location target encoding
    train_loc_enc, test_loc_enc = smoothed_target_encoding(
        train_records["location"],
        train_records["connection_time"],
        test_records["location"],
    )
    train_feats["location_target_enc"] = train_loc_enc
    test_feats["location_target_enc"] = test_loc_enc

    # 3. Overnight Stage-1 classifier: P(overnight session > 8.0h)
    train_is_overnight = (train_records["connection_time"] > 8.0).astype(int).to_numpy()
    stage1_cols = [
        "plugin_hour_sin",
        "plugin_hour_cos",
        "plugin_weekday_sin",
        "plugin_weekday_cos",
        "plugin_weekend",
        "next_day_is_workday",
        "nhts_p_home_arrival",
        "nhts_p_first_home_departure",
        "nhts_expected_home_stay_hours",
        "history_count",
        "hours_since_last_plugout",
        "user_hist_median_conn",
        "user_hist_weekday_median_conn",
        "user_hist_weekend_median_conn",
        "location_target_enc",
    ]
    x_tr_stage1 = train_feats[stage1_cols].to_numpy()
    x_te_stage1 = test_feats[stage1_cols].to_numpy()

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_p_overnight = np.zeros(len(train_records), dtype=float)
    for tr_idx, val_idx in kf.split(x_tr_stage1):
        c_fold = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.08, max_leaf_nodes=15, random_state=RANDOM_SEED
        ).fit(x_tr_stage1[tr_idx], train_is_overnight[tr_idx])
        oof_p_overnight[val_idx] = c_fold.predict_proba(x_tr_stage1[val_idx])[:, 1]

    clf_full = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.08, max_leaf_nodes=15, random_state=RANDOM_SEED
    ).fit(x_tr_stage1, train_is_overnight)
    test_p_overnight = clf_full.predict_proba(x_te_stage1)[:, 1]

    train_feats["p_overnight"] = oof_p_overnight
    test_feats["p_overnight"] = test_p_overnight

    # Exclude non-feature column
    feature_cols = [c for c in train_feats.columns if c != "causal_user_baseline"]
    X_train = train_feats[feature_cols].to_numpy()
    y_train = train_records["connection_time"].to_numpy()
    X_test = test_feats[feature_cols].to_numpy()
    y_test = test_records["connection_time"].to_numpy()

    # 4. Point estimate regressor (L1 / absolute error loss)
    point_model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    ).fit(X_train, y_train)
    point_pred = np.maximum(0.1, point_model.predict(X_test))

    # 5. Quantile regressors (p10, p50, p90)
    q10_model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.10,
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
    ).fit(X_train, y_train)
    p10_pred = np.maximum(0.05, q10_model.predict(X_test))

    q50_model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.50,
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
    ).fit(X_train, y_train)
    p50_pred = np.maximum(0.1, q50_model.predict(X_test))

    q90_model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.90,
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
    ).fit(X_train, y_train)
    p90_pred = np.maximum(p50_pred, q90_model.predict(X_test))

    # 6. Baselines
    tr_median = float(np.median(y_train))
    tr_p10 = float(np.quantile(y_train, 0.10))
    tr_p90 = float(np.quantile(y_train, 0.90))

    median_baseline = np.full(len(y_test), tr_median)
    history_baseline = np.where(
        test_feats["history_count"].to_numpy() > 0,
        test_feats["causal_user_baseline"].to_numpy(),
        tr_median,
    )

    seen_users = set(train_records["user_id"].astype(str))
    warm_mask = test_records["user_id"].astype(str).isin(seen_users).to_numpy()

    return TargetEvaluation(
        target_name=TARGET_CONNECTION,
        evaluation_name=evaluation_name,
        train_x=train_feats[feature_cols],
        train_y=train_records["connection_time"],
        test_x=test_feats[feature_cols],
        test_y=test_records["connection_time"],
        test_records=test_records,
        test_users=test_records["user_id"],
        warm_mask=warm_mask,
        point_pred=point_pred,
        p10_pred=p10_pred,
        p50_pred=p50_pred,
        p90_pred=p90_pred,
        median_baseline=median_baseline,
        history_baseline=history_baseline,
        baseline_p10=np.full(len(y_test), tr_p10),
        baseline_p90=np.full(len(y_test), tr_p90),
    )


def fit_and_evaluate_next_gap(
    train_records: pd.DataFrame,
    test_records: pd.DataFrame,
    priors: NHTSPriors,
    evaluation_name: str,
    allow_test_history: bool,
) -> TargetEvaluation:
    """Fit point and quantile regressors for next_gap_hours."""
    # Filter rows with valid next_gap_hours (exclude user's terminal session)
    tr_valid = train_records.loc[train_records["next_gap_hours"].notna()].copy()
    te_valid = test_records.loc[test_records["next_gap_hours"].notna()].copy()

    # 1. Feature extraction
    train_feats, train_histories, fb_stats = extract_next_gap_features(
        tr_valid, priors, update_history=True
    )
    test_feats, _, _ = extract_next_gap_features(
        te_valid,
        priors,
        initial_histories=train_histories,
        update_history=allow_test_history,
        fallback_stats=fb_stats,
    )

    # 2. Location target encoding
    train_loc_enc, test_loc_enc = smoothed_target_encoding(
        tr_valid["location"], tr_valid["next_gap_hours"], te_valid["location"]
    )
    train_feats["location_target_enc"] = train_loc_enc
    test_feats["location_target_enc"] = test_loc_enc

    feature_cols = [c for c in train_feats.columns if c != "causal_user_gap_baseline"]
    X_train = train_feats[feature_cols].to_numpy()
    y_train = tr_valid["next_gap_hours"].to_numpy()
    X_test = test_feats[feature_cols].to_numpy()
    y_test = te_valid["next_gap_hours"].to_numpy()

    # 3. Point estimate regressor (L1 loss)
    point_model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    ).fit(X_train, y_train)
    point_pred = np.maximum(0.1, point_model.predict(X_test))

    # 4. Quantile regressors (p10, p50, p90)
    q10_model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.10,
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
    ).fit(X_train, y_train)
    p10_pred = np.maximum(0.05, q10_model.predict(X_test))

    q50_model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.50,
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
    ).fit(X_train, y_train)
    p50_pred = np.maximum(0.1, q50_model.predict(X_test))

    q90_model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.90,
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
    ).fit(X_train, y_train)
    p90_pred = np.maximum(p50_pred, q90_model.predict(X_test))

    # 5. Baselines
    tr_median = float(np.median(y_train))
    tr_p10 = float(np.quantile(y_train, 0.10))
    tr_p90 = float(np.quantile(y_train, 0.90))

    median_baseline = np.full(len(y_test), tr_median)
    history_baseline = np.where(
        test_feats["history_count"].to_numpy() > 0,
        test_feats["causal_user_gap_baseline"].to_numpy(),
        tr_median,
    )

    seen_users = set(tr_valid["user_id"].astype(str))
    warm_mask = te_valid["user_id"].astype(str).isin(seen_users).to_numpy()

    return TargetEvaluation(
        target_name=TARGET_GAP,
        evaluation_name=evaluation_name,
        train_x=train_feats[feature_cols],
        train_y=tr_valid["next_gap_hours"],
        test_x=test_feats[feature_cols],
        test_y=te_valid["next_gap_hours"],
        test_records=te_valid,
        test_users=te_valid["user_id"],
        warm_mask=warm_mask,
        point_pred=point_pred,
        p10_pred=p10_pred,
        p50_pred=p50_pred,
        p90_pred=p90_pred,
        median_baseline=median_baseline,
        history_baseline=history_baseline,
        baseline_p10=np.full(len(y_test), tr_p10),
        baseline_p90=np.full(len(y_test), tr_p90),
    )


def compute_metrics_table(evaluations: List[TargetEvaluation]) -> pd.DataFrame:
    """Compute comprehensive performance metrics across models, targets, and evaluation splits."""
    rows: List[Dict[str, object]] = []

    for ev in evaluations:
        y_true = ev.test_y.to_numpy()
        subsets = {
            "all_test": np.ones(len(y_true), dtype=bool),
        }
        if "chronological" in ev.evaluation_name:
            subsets["warm_user"] = ev.warm_mask
            subsets["cold_start_user"] = ~ev.warm_mask

        for subset_name, mask in subsets.items():
            if not np.any(mask):
                continue
            sub_y = y_true[mask]
            sub_recs = ev.test_records.iloc[mask]
            n_sess = int(mask.sum())
            n_users = int(ev.test_users.iloc[mask].nunique())

            # Evaluate each predictor
            predictors = {
                "training_median": ev.median_baseline[mask],
                "causal_user_history_median": ev.history_baseline[mask],
                "hist_gradient_boosting": ev.point_pred[mask],
            }

            for pred_name, y_pred in predictors.items():
                mae = mean_absolute_error(sub_y, y_pred)
                rmse = root_mean_squared_error(sub_y, y_pred)

                # Recovered clock-hour error
                if ev.target_name == TARGET_CONNECTION:
                    # Connection time: plugout_time = plugin_time + connection_time
                    plugin_dts = sub_recs["plugin_dt"]
                    true_plugout_dts = sub_recs["plugout_dt"]
                    pred_plugout_dts = plugin_dts + pd.to_timedelta(y_pred, unit="h")

                    true_h = true_plugout_dts.dt.hour + true_plugout_dts.dt.minute / 60.0
                    pred_h = pred_plugout_dts.dt.hour + pred_plugout_dts.dt.minute / 60.0
                    circ_hour_mae = _circular_hour_mae(true_h.to_numpy(), pred_h.to_numpy())
                else:
                    # Next gap: next_plugin_time = plugout_time + next_gap_hours
                    plugout_dts = sub_recs["plugout_dt"]
                    true_next_in_dts = sub_recs["next_plugin_dt"]
                    pred_next_in_dts = plugout_dts + pd.to_timedelta(y_pred, unit="h")

                    true_h = true_next_in_dts.dt.hour + true_next_in_dts.dt.minute / 60.0
                    pred_h = pred_next_in_dts.dt.hour + pred_next_in_dts.dt.minute / 60.0
                    circ_hour_mae = _circular_hour_mae(true_h.to_numpy(), pred_h.to_numpy())

                # Quantile metrics
                if pred_name == "hist_gradient_boosting":
                    sub_p10 = ev.p10_pred[mask]
                    sub_p50 = ev.p50_pred[mask]
                    sub_p90 = ev.p90_pred[mask]
                else:
                    sub_p10 = ev.baseline_p10[mask]
                    sub_p50 = y_pred
                    sub_p90 = ev.baseline_p90[mask]

                pb10 = _pinball_loss(sub_y, sub_p10, 0.10)
                pb50 = _pinball_loss(sub_y, sub_p50, 0.50)
                pb90 = _pinball_loss(sub_y, sub_p90, 0.90)
                cov80, width = _interval_metrics(sub_y, sub_p10, sub_p90)

                rows.append(
                    {
                        "target": ev.target_name,
                        "evaluation": ev.evaluation_name,
                        "subset": subset_name,
                        "predictor": pred_name,
                        "n_sessions": n_sess,
                        "n_users": n_users,
                        "mae_hours": mae,
                        "rmse_hours": rmse,
                        "circular_hour_mae": circ_hour_mae,
                        "pinball_loss_p10": pb10,
                        "pinball_loss_p50": pb50,
                        "pinball_loss_p90": pb90,
                        "interval_coverage_80": cov80,
                        "mean_interval_width_hours": width,
                    }
                )

    return pd.DataFrame(rows)


def save_diagnostic_plots(
    conn_eval: TargetEvaluation,
    gap_eval: TargetEvaluation,
    output_dir: Path,
) -> None:
    """Generate and save presentation-quality diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Connection Time: Actual vs Predicted + Quantiles
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    recs = conn_eval.test_records.copy()
    y_true = conn_eval.test_y.to_numpy()
    y_pred = conn_eval.point_pred
    p10 = conn_eval.p10_pred
    p90 = conn_eval.p90_pred
    warm = conn_eval.warm_mask

    # Sample up to 2500 for clean visualization
    np.random.seed(RANDOM_SEED)
    idx_sample = np.random.choice(len(y_true), size=min(2500, len(y_true)), replace=False)

    ax.scatter(
        y_true[idx_sample][warm[idx_sample]],
        y_pred[idx_sample][warm[idx_sample]],
        s=14,
        alpha=0.35,
        color="#1f77b4",
        label="Warm users (seen in training)",
        edgecolors="none",
    )
    ax.scatter(
        y_true[idx_sample][~warm[idx_sample]],
        y_pred[idx_sample][~warm[idx_sample]],
        s=14,
        alpha=0.35,
        color="#ff7f0e",
        label="Cold-start users (unseen in training)",
        edgecolors="none",
    )

    limit = 48.0  # Show up to 48 hours for clear view of daytime vs overnight
    ax.plot([0, limit], [0, limit], color="#d62728", linestyle="--", linewidth=1.8, label="Ideal prediction (1:1)")
    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="Observed Connection Time (hours)",
        ylabel="Predicted Connection Time (hours)",
        title="Future Chronological Split: Plug-out Duration (Connection Time)",
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "ml_connection_time_actual_vs_predicted.png", dpi=200)
    plt.close(fig)

    # 2. Next Gap Hours: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    y_true_g = gap_eval.test_y.to_numpy()
    y_pred_g = gap_eval.point_pred
    warm_g = gap_eval.warm_mask

    idx_sample_g = np.random.choice(len(y_true_g), size=min(2500, len(y_true_g)), replace=False)
    ax.scatter(
        y_true_g[idx_sample_g][warm_g[idx_sample_g]],
        y_pred_g[idx_sample_g][warm_g[idx_sample_g]],
        s=14,
        alpha=0.35,
        color="#2ca02c",
        label="Warm users",
        edgecolors="none",
    )
    ax.scatter(
        y_true_g[idx_sample_g][~warm_g[idx_sample_g]],
        y_pred_g[idx_sample_g][~warm_g[idx_sample_g]],
        s=14,
        alpha=0.35,
        color="#9467bd",
        label="Cold-start users",
        edgecolors="none",
    )

    limit_g = 120.0  # Show up to 5 days
    ax.plot([0, limit_g], [0, limit_g], color="#d62728", linestyle="--", linewidth=1.8, label="Ideal prediction (1:1)")
    ax.set(
        xlim=(0, limit_g),
        ylim=(0, limit_g),
        xlabel="Observed Next Gap Hours (hours to next session)",
        ylabel="Predicted Next Gap Hours",
        title="Future Chronological Split: Next Plug-in Timing (Gap Hours)",
    )
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "ml_next_gap_actual_vs_predicted.png", dpi=200)
    plt.close(fig)

    # 3. Diagnostic by Day of Week: Connection Time and Next Gap intervals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_conn = conn_eval.test_records["plugin_dt"].dt.dayofweek.to_numpy()
    dow_gap = gap_eval.test_records["plugout_dt"].dt.dayofweek.to_numpy()

    # Connection time by day of week
    med_actual_conn = [np.median(y_true[dow_conn == d]) for d in range(7)]
    med_pred_conn = [np.median(y_pred[dow_conn == d]) for d in range(7)]
    p10_dow_conn = [np.median(p10[dow_conn == d]) for d in range(7)]
    p90_dow_conn = [np.median(p90[dow_conn == d]) for d in range(7)]

    x = np.arange(7)
    ax1.plot(x, med_actual_conn, marker="o", color="#1f77b4", linewidth=2.0, label="Observed Median")
    ax1.plot(x, med_pred_conn, marker="s", color="#d62728", linestyle="--", linewidth=2.0, label="Predicted Median (p50)")
    ax1.fill_between(x, p10_dow_conn, p90_dow_conn, color="#1f77b4", alpha=0.18, label="Predicted [p10, p90] Interval")
    ax1.set_xticks(x)
    ax1.set_xticklabels(days)
    ax1.set(
        xlabel="Plug-in Day of Week",
        ylabel="Connection Time (hours)",
        title="Connection Time Profile by Day of Week",
        ylim=(0, 24),
    )
    ax1.legend(frameon=True, facecolor="white", loc="lower right")

    # Next gap by day of week
    med_actual_gap = [np.median(y_true_g[dow_gap == d]) for d in range(7)]
    med_pred_gap = [np.median(y_pred_g[dow_gap == d]) for d in range(7)]
    p10_dow_gap = [np.median(gap_eval.p10_pred[dow_gap == d]) for d in range(7)]
    p90_dow_gap = [np.median(gap_eval.p90_pred[dow_gap == d]) for d in range(7)]

    ax2.plot(x, med_actual_gap, marker="o", color="#2ca02c", linewidth=2.0, label="Observed Median")
    ax2.plot(x, med_pred_gap, marker="s", color="#e377c2", linestyle="--", linewidth=2.0, label="Predicted Median (p50)")
    ax2.fill_between(x, p10_dow_gap, p90_dow_gap, color="#2ca02c", alpha=0.18, label="Predicted [p10, p90] Interval")
    ax2.set_xticks(x)
    ax2.set_xticklabels(days)
    ax2.set(
        xlabel="Plug-out Day of Week",
        ylabel="Next Gap Hours (hours)",
        title="Inter-Session Gap Profile by Day of Week",
        ylim=(0, 48),
    )
    ax2.legend(frameon=True, facecolor="white", loc="upper left")

    fig.suptitle("EV Charging Timing Profiles and Quantile Envelopes across Days of the Week", fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / "ml_charging_times_by_dayofweek.png", dpi=200)
    plt.close(fig)


def run_experiment(root: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, object]]:
    """Run full chronological and unseen-user evaluations for both connection_time and next_gap_hours."""
    raw_path = root / "data" / "raw" / "norway_ev" / "Dataset1_charging_reports.csv"
    chains_path = root / "data" / "interim" / "vehicle_trip_chains.csv"
    park_path = root / "data" / "interim" / "vehicle_parking_intervals.csv"

    sessions = load_clean_charging_data(raw_path)
    priors = build_nhts_priors(chains_path, park_path)

    # 1. Chronological split (80/20)
    cutoff_index = int(len(sessions) * 0.8)
    cutoff = sessions.loc[cutoff_index, "plugin_dt"]
    temporal_train = sessions.loc[sessions["plugin_dt"] < cutoff].copy().reset_index(drop=True)
    temporal_test = sessions.loc[sessions["plugin_dt"] >= cutoff].copy().reset_index(drop=True)

    # Fit & evaluate connection_time (future chronological)
    conn_eval_temporal = fit_and_evaluate_connection_time(
        temporal_train, temporal_test, priors, "future_session_chronological", allow_test_history=True
    )

    # Fit & evaluate next_gap_hours (future chronological)
    gap_eval_temporal = fit_and_evaluate_next_gap(
        temporal_train, temporal_test, priors, "future_session_chronological", allow_test_history=True
    )

    # 2. Unseen-user holdout split (GroupShuffleSplit 80/20 on user_id)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    group_train_idx, group_test_idx = next(splitter.split(sessions, groups=sessions["user_id"]))
    group_train = sessions.iloc[group_train_idx].copy().reset_index(drop=True)
    group_test = sessions.iloc[group_test_idx].copy().reset_index(drop=True)

    # Fit & evaluate connection_time (unseen user)
    conn_eval_group = fit_and_evaluate_connection_time(
        group_train, group_test, priors, "unseen_user_group_holdout", allow_test_history=False
    )

    # Fit & evaluate next_gap_hours (unseen user)
    gap_eval_group = fit_and_evaluate_next_gap(
        group_train, group_test, priors, "unseen_user_group_holdout", allow_test_history=False
    )

    # Compute comprehensive metrics table
    metrics_df = compute_metrics_table(
        [conn_eval_temporal, gap_eval_temporal, conn_eval_group, gap_eval_group]
    )

    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = outputs_dir / "ml_charging_times_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False, float_format="%.4f")

    # Save diagnostic figures
    save_diagnostic_plots(conn_eval_temporal, gap_eval_temporal, outputs_dir)

    # Prepare prediction tables for notebook inspection
    prediction_tables = {
        "connection_time_temporal": pd.DataFrame(
            {
                "plugin_dt": conn_eval_temporal.test_records["plugin_dt"].to_numpy(),
                "user_id": conn_eval_temporal.test_records["user_id"].to_numpy(),
                "actual_connection_hours": conn_eval_temporal.test_y.to_numpy(),
                "predicted_connection_hours": conn_eval_temporal.point_pred,
                "p10_hours": conn_eval_temporal.p10_pred,
                "p50_hours": conn_eval_temporal.p50_pred,
                "p90_hours": conn_eval_temporal.p90_pred,
                "warm_user": conn_eval_temporal.warm_mask,
            }
        ),
        "next_gap_temporal": pd.DataFrame(
            {
                "plugout_dt": gap_eval_temporal.test_records["plugout_dt"].to_numpy(),
                "user_id": gap_eval_temporal.test_records["user_id"].to_numpy(),
                "actual_gap_hours": gap_eval_temporal.test_y.to_numpy(),
                "predicted_gap_hours": gap_eval_temporal.point_pred,
                "p10_hours": gap_eval_temporal.p10_pred,
                "p50_hours": gap_eval_temporal.p50_pred,
                "p90_hours": gap_eval_temporal.p90_pred,
                "warm_user": gap_eval_temporal.warm_mask,
            }
        ),
    }

    metadata = {
        "temporal_cutoff": str(cutoff),
        "temporal_train_sessions": len(temporal_train),
        "temporal_test_sessions": len(temporal_test),
        "temporal_train_users": temporal_train["user_id"].nunique(),
        "temporal_test_users": temporal_test["user_id"].nunique(),
        "temporal_warm_sessions": int(conn_eval_temporal.warm_mask.sum()),
        "temporal_cold_sessions": int((~conn_eval_temporal.warm_mask).sum()),
        "group_train_sessions": len(group_train),
        "group_test_sessions": len(group_test),
        "group_train_users": group_train["user_id"].nunique(),
        "group_test_users": group_test["user_id"].nunique(),
    }

    return metrics_df, prediction_tables, metadata


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    metrics, _, meta = run_experiment(project_root)
    print("=== Metadata ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print("\n=== Evaluation Metrics ===")
    print(metrics.to_string(index=False))
