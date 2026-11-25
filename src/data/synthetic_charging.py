"""Synthetic EV charging dataset generator.

Creates a realistic synthetic dataset matching the exact schema and format of
``data/raw/norway_ev/Dataset1_charging_reports.csv`` (semicolon-separated,
comma-decimal) so that ``src/models/charging_times.py`` can be exercised on
data with *known* ground-truth behavioural archetypes.

Archetypes
----------
1. Rigid Weekday Commuters (~40 %)
   Plug in ~17:30-18:30 on workdays; unplug ~07:30-08:15 next workday morning.
   Weekend/holiday sessions shift later and are shorter.

2. Multi-Day / Energy-Deficit Chargers (~30 %)
   Drive a daily kWh burn rate; plug in every 2-3 days once accumulated energy
   deficit crosses a threshold.

3. Mixed Daytime & Overnight Users (~20 %)
   Alternate between short daytime top-ups (1.5-4 h) and longer overnight stays
   (>10 h).

4. Late-Joining / Cold-Start Users (~10 %)
   Appear only in the final 20 % of the timeline.  Assigned one of the above
   behavioural patterns for variety.

Edge Cases
----------
- ~0.2 % of rows have missing ``plugout_time`` / ``connection_time``.
- Gaussian clock jitter on plugin/plugout times.
- ``energy_session`` proportional to ``connection_time`` capped by
  battery/charger limits.

Reproducibility
---------------
All random draws are seeded via ``RANDOM_SEED`` (default 42).
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

import holidays
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — all tunables are gathered here, never hardcoded inline.
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

# Timeline
TIMELINE_START = pd.Timestamp("2019-01-01")
TIMELINE_END = pd.Timestamp("2019-12-31 23:59:59")

# User pool
N_LOCATIONS = 6
LOCATION_NAMES = [f"LOC_{chr(65 + i)}" for i in range(N_LOCATIONS)]  # LOC_A .. LOC_F
N_COMMUTERS = 48       # ~40 %
N_MULTIDAY = 36        # ~30 %
N_MIXED = 24           # ~20 %
N_COLD_START = 12      # ~10 %
N_TOTAL = N_COMMUTERS + N_MULTIDAY + N_MIXED + N_COLD_START  # 120

# Charger / battery
DEFAULT_CHARGER_KW = 7.4        # Typical Level-2 home charger
DEFAULT_BATTERY_KWH = 60.0
CHARGING_EFFICIENCY = 0.92
MIN_SOC_FRAC = 0.10             # 10 %
TARGET_SOC_FRAC = 0.90          # 90 %

# Missing-data fraction
MISSING_FRACTION = 0.002        # 0.2 %

# Norwegian public holidays (same range as the pipeline uses)
NO_HOLIDAYS = set(holidays.country_holidays("NO", years=range(2018, 2023)).keys())


def _is_workday(d: pd.Timestamp) -> bool:
    """Return True when *d* is a weekday (Mon-Fri) that is not a Norwegian public holiday."""
    return d.weekday() < 5 and d.date() not in NO_HOLIDAYS


def _next_workday(d: pd.Timestamp) -> pd.Timestamp:
    """Return the next calendar day that is a workday."""
    nxt = d.normalize() + pd.Timedelta(days=1)
    while not _is_workday(nxt):
        nxt += pd.Timedelta(days=1)
    return nxt


def _clamp_ts(ts: pd.Timestamp) -> pd.Timestamp:
    """Clamp to whole minutes (remove seconds for clean formatting)."""
    return ts.floor("min")


# ---------------------------------------------------------------------------
# Per-archetype session generators
# ---------------------------------------------------------------------------

def _generate_commuter_sessions(
    rng: np.random.Generator,
    user_id: str,
    location: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[dict]:
    """Rigid weekday commuter: plug in ~17:30-18:30 on workdays, unplug next workday morning ~07:30-08:15."""
    sessions: List[dict] = []
    day = start.normalize()
    while day <= end:
        if _is_workday(day):
            # --- Workday session ---
            plugin_hour = 17.5 + rng.normal(0.25, 0.25)  # ~17:30 ± 15 min
            plugin_hour = np.clip(plugin_hour, 16.5, 20.0)
            plugin_dt = _clamp_ts(day + pd.Timedelta(hours=plugin_hour))

            next_wd = _next_workday(day)
            plugout_hour = 7.5 + rng.normal(0.25, 0.12)  # ~07:30-08:15
            plugout_hour = np.clip(plugout_hour, 6.5, 9.0)
            plugout_dt = _clamp_ts(next_wd + pd.Timedelta(hours=plugout_hour))

            conn_hours = (plugout_dt - plugin_dt).total_seconds() / 3600.0
            energy = min(
                DEFAULT_CHARGER_KW * CHARGING_EFFICIENCY * conn_hours,
                DEFAULT_BATTERY_KWH * (TARGET_SOC_FRAC - MIN_SOC_FRAC),
            ) * rng.uniform(0.65, 1.0)

            sessions.append(
                {
                    "location": location,
                    "user_id": user_id,
                    "plugin_time": plugin_dt,
                    "plugout_time": plugout_dt,
                    "connection_time": round(conn_hours, 5),
                    "energy_session": round(max(0.5, energy), 2),
                }
            )
        else:
            # Weekend / holiday: ~60 % chance of a shorter session
            if rng.random() < 0.60:
                plugin_hour = rng.uniform(19.0, 21.0)
                plugin_dt = _clamp_ts(day + pd.Timedelta(hours=plugin_hour))
                conn_hours = rng.uniform(5.0, 12.0)
                plugout_dt = _clamp_ts(plugin_dt + pd.Timedelta(hours=conn_hours))
                energy = min(
                    DEFAULT_CHARGER_KW * CHARGING_EFFICIENCY * conn_hours,
                    DEFAULT_BATTERY_KWH * (TARGET_SOC_FRAC - MIN_SOC_FRAC),
                ) * rng.uniform(0.50, 0.95)
                sessions.append(
                    {
                        "location": location,
                        "user_id": user_id,
                        "plugin_time": plugin_dt,
                        "plugout_time": plugout_dt,
                        "connection_time": round(conn_hours, 5),
                        "energy_session": round(max(0.5, energy), 2),
                    }
                )

        day += pd.Timedelta(days=1)

    return sessions


def _generate_multiday_sessions(
    rng: np.random.Generator,
    user_id: str,
    location: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[dict]:
    """Multi-day / energy-deficit charger: plug in every 2-3 days once deficit exceeds threshold."""
    sessions: List[dict] = []
    daily_burn = rng.uniform(8.0, 15.0)
    soc = rng.uniform(0.70, 0.90) * DEFAULT_BATTERY_KWH
    threshold_kwh = rng.uniform(0.30, 0.40) * DEFAULT_BATTERY_KWH

    day = start.normalize()
    while day <= end:
        soc -= daily_burn * rng.uniform(0.8, 1.2)
        if soc <= threshold_kwh:
            plugin_hour = rng.uniform(17.0, 21.0)
            plugin_dt = _clamp_ts(day + pd.Timedelta(hours=plugin_hour))
            conn_hours = rng.uniform(8.0, 16.0)
            plugout_dt = _clamp_ts(plugin_dt + pd.Timedelta(hours=conn_hours))

            charged = DEFAULT_CHARGER_KW * CHARGING_EFFICIENCY * conn_hours
            energy = min(charged, DEFAULT_BATTERY_KWH * TARGET_SOC_FRAC - soc)
            energy = max(0.5, energy * rng.uniform(0.80, 1.0))

            sessions.append(
                {
                    "location": location,
                    "user_id": user_id,
                    "plugin_time": plugin_dt,
                    "plugout_time": plugout_dt,
                    "connection_time": round(conn_hours, 5),
                    "energy_session": round(energy, 2),
                }
            )
            soc = min(DEFAULT_BATTERY_KWH * TARGET_SOC_FRAC, soc + charged)
        day += pd.Timedelta(days=1)

    return sessions


def _generate_mixed_sessions(
    rng: np.random.Generator,
    user_id: str,
    location: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[dict]:
    """Mixed daytime & overnight: alternate short top-ups (1.5-4 h) and overnight stays (>10 h)."""
    sessions: List[dict] = []
    day = start.normalize()
    overnight_next = bool(rng.integers(2))  # alternate starting state

    while day <= end:
        do_session = rng.random() < 0.70  # ~4-5 days/week
        if do_session:
            if overnight_next:
                plugin_hour = rng.uniform(18.0, 22.0)
                conn_hours = rng.uniform(10.0, 16.0)
            else:
                plugin_hour = rng.uniform(10.0, 14.0)
                conn_hours = rng.uniform(1.5, 4.0)

            plugin_dt = _clamp_ts(day + pd.Timedelta(hours=plugin_hour))
            plugout_dt = _clamp_ts(plugin_dt + pd.Timedelta(hours=conn_hours))

            energy = min(
                DEFAULT_CHARGER_KW * CHARGING_EFFICIENCY * conn_hours,
                DEFAULT_BATTERY_KWH * (TARGET_SOC_FRAC - MIN_SOC_FRAC),
            ) * rng.uniform(0.55, 1.0)

            sessions.append(
                {
                    "location": location,
                    "user_id": user_id,
                    "plugin_time": plugin_dt,
                    "plugout_time": plugout_dt,
                    "connection_time": round(conn_hours, 5),
                    "energy_session": round(max(0.5, energy), 2),
                }
            )
            overnight_next = not overnight_next  # alternate
        day += pd.Timedelta(days=1)

    return sessions


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    seed: int = RANDOM_SEED,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate a full synthetic dataset and optionally write to CSV.

    Parameters
    ----------
    seed : int
        Random seed for full reproducibility.
    output_path : Path or None
        If given, write the CSV matching Norway Dataset 1 format.

    Returns
    -------
    pd.DataFrame
        DataFrame of generated sessions (before introducing missing values).
    """
    rng = np.random.default_rng(seed)

    all_sessions: List[dict] = []
    session_counter = 0
    user_counter = 0

    # Assign locations round-robin across users
    def _loc(uid_idx: int) -> str:
        return LOCATION_NAMES[uid_idx % N_LOCATIONS]

    # --- 1. Rigid Weekday Commuters ---
    for i in range(N_COMMUTERS):
        uid = f"SYN_COMMUTER_{i:03d}"
        loc = _loc(user_counter)
        sess = _generate_commuter_sessions(rng, uid, loc, TIMELINE_START, TIMELINE_END)
        all_sessions.extend(sess)
        user_counter += 1

    # --- 2. Multi-Day / Energy-Deficit Chargers ---
    for i in range(N_MULTIDAY):
        uid = f"SYN_MULTIDAY_{i:03d}"
        loc = _loc(user_counter)
        sess = _generate_multiday_sessions(rng, uid, loc, TIMELINE_START, TIMELINE_END)
        all_sessions.extend(sess)
        user_counter += 1

    # --- 3. Mixed Daytime & Overnight ---
    for i in range(N_MIXED):
        uid = f"SYN_MIXED_{i:03d}"
        loc = _loc(user_counter)
        sess = _generate_mixed_sessions(rng, uid, loc, TIMELINE_START, TIMELINE_END)
        all_sessions.extend(sess)
        user_counter += 1

    # --- 4. Cold-Start / Late-Joining Users (final 20 % of timeline) ---
    cold_start_begin = TIMELINE_START + (TIMELINE_END - TIMELINE_START) * 0.80
    generators = [_generate_commuter_sessions, _generate_multiday_sessions, _generate_mixed_sessions]
    for i in range(N_COLD_START):
        uid = f"SYN_COLD_{i:03d}"
        loc = _loc(user_counter)
        gen_fn = generators[i % len(generators)]
        sess = gen_fn(rng, uid, loc, cold_start_begin, TIMELINE_END)
        all_sessions.extend(sess)
        user_counter += 1

    # Assign session IDs
    df = pd.DataFrame(all_sessions)
    df = df.sort_values("plugin_time").reset_index(drop=True)
    df["session_id"] = [f"SYN_{i:06d}" for i in range(len(df))]

    # --- Introduce missing values (~0.2 %) ---
    n_missing = max(1, int(len(df) * MISSING_FRACTION))
    missing_idx = rng.choice(len(df), size=n_missing, replace=False)
    df.loc[missing_idx, "plugout_time"] = pd.NaT
    df.loc[missing_idx, "connection_time"] = np.nan

    # --- Write to CSV matching Norway format ---
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_norway_format_csv(df, output_path)

    return df


def _write_norway_format_csv(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV matching Norway Dataset 1 format exactly.

    Format rules:
    - Semicolon separator
    - String columns are quoted
    - Datetimes as ``YYYY-MM-DD HH:MM:SS`` (no quotes)
    - Floats use comma as decimal separator
    - Missing plugout_time/connection_time are empty strings
    """
    col_order = ["location", "user_id", "session_id", "plugin_time", "plugout_time", "connection_time", "energy_session"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        # Header with quoted names
        f.write(";".join(f'"{c}"' for c in col_order) + "\n")
        for _, row in df.iterrows():
            parts: List[str] = []
            for c in col_order:
                val = row[c]
                if c in ("location", "user_id", "session_id"):
                    parts.append(f'"{val}"')
                elif c in ("plugin_time", "plugout_time"):
                    if pd.isna(val):
                        parts.append("")
                    else:
                        parts.append(val.strftime("%Y-%m-%d %H:%M:%S"))
                elif c in ("connection_time", "energy_session"):
                    if pd.isna(val):
                        parts.append("")
                    else:
                        # Comma as decimal separator
                        parts.append(str(round(float(val), 5)).replace(".", ","))
                else:
                    parts.append(str(val))
            f.write(";".join(parts) + "\n")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    out = project_root / "data" / "synthetic" / "synthetic_charging_reports.csv"
    df = generate_synthetic_dataset(output_path=out)
    print(f"Generated {len(df)} sessions for {df['user_id'].nunique()} users")
    print(f"  Locations : {sorted(df['location'].unique())}")
    print(f"  Date range: {df['plugin_time'].min()} to {df['plugin_time'].max()}")
    print(f"  Missing   : {df['plugout_time'].isna().sum()} rows with missing plugout_time")
    print(f"  Saved to  : {out}")
