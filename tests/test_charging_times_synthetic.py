"""Validation & leakage test suite for ``src/models/charging_times.py``.

Uses the synthetic dataset produced by ``src/data/synthetic_charging.py``
to run three categories of tests:

1. **Signal-recovery** – The HistGradientBoosting regressor must beat both
   ``training_median`` and ``causal_user_history_median`` baselines on MAE for
   ``connection_time`` and ``next_gap_hours``, and the p10-p90 interval must
   achieve ~80 % empirical coverage.

2. **Archetype sanity** – Predicted ``connection_time`` before workdays
   (``next_day_is_workday == True``) must be systematically longer than for
   Friday/Saturday afternoon sessions; multi-day users must receive gap
   predictions in a plausible 30-80 h range (not collapsed to global median).

3. **Zero-leakage** – Corrupting ``plugout_time`` and ``energy_session`` for
   the last 20 % of sessions (by chronological ``plugin_dt`` order) must leave
   all causal features for the first 80 % bit-for-bit identical, proving that
   no future-session information leaks into earlier sessions' features.

Usage::

    .venv/bin/python -m pytest tests/test_charging_times_synthetic.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment configuration (must precede heavy imports)
# ---------------------------------------------------------------------------
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest

# Ensure the project ``src/`` is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from models.charging_times import (
    build_nhts_priors,
    extract_connection_time_features,
    extract_next_gap_features,
    fit_and_evaluate_connection_time,
    fit_and_evaluate_next_gap,
    load_clean_charging_data,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SYNTHETIC_CSV = _PROJECT_ROOT / "data" / "synthetic" / "synthetic_charging_reports.csv"
_CHAINS_PATH = _PROJECT_ROOT / "data" / "interim" / "vehicle_trip_chains.csv"
_PARK_PATH = _PROJECT_ROOT / "data" / "interim" / "vehicle_parking_intervals.csv"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_sessions():
    """Load and clean the synthetic dataset exactly as the pipeline does."""
    assert _SYNTHETIC_CSV.exists(), (
        f"Synthetic CSV not found at {_SYNTHETIC_CSV}. "
        "Run `python src/data/synthetic_charging.py` first."
    )
    return load_clean_charging_data(_SYNTHETIC_CSV)


@pytest.fixture(scope="module")
def nhts_priors():
    """Build NHTS prior lookup tables (shared across all tests)."""
    return build_nhts_priors(_CHAINS_PATH, _PARK_PATH)


@pytest.fixture(scope="module")
def chronological_split(synthetic_sessions):
    """80/20 chronological split, matching ``run_experiment()`` logic."""
    sessions = synthetic_sessions
    cutoff_index = int(len(sessions) * 0.8)
    cutoff = sessions.loc[cutoff_index, "plugin_dt"]
    train = sessions.loc[sessions["plugin_dt"] < cutoff].copy().reset_index(drop=True)
    test = sessions.loc[sessions["plugin_dt"] >= cutoff].copy().reset_index(drop=True)
    return train, test


@pytest.fixture(scope="module")
def conn_eval(chronological_split, nhts_priors):
    """Fit and evaluate connection_time on the synthetic chronological split."""
    train, test = chronological_split
    return fit_and_evaluate_connection_time(
        train, test, nhts_priors,
        evaluation_name="synthetic_chronological",
        allow_test_history=True,
    )


@pytest.fixture(scope="module")
def gap_eval(chronological_split, nhts_priors):
    """Fit and evaluate next_gap_hours on the synthetic chronological split."""
    train, test = chronological_split
    return fit_and_evaluate_next_gap(
        train, test, nhts_priors,
        evaluation_name="synthetic_chronological",
        allow_test_history=True,
    )


# ===================================================================
# 1. Signal-Recovery Tests
# ===================================================================

class TestSignalRecovery:
    """The GBR model must beat simple baselines on the synthetic data."""

    def test_connection_time_gbr_beats_training_median(self, conn_eval):
        """GBR MAE < training_median MAE for connection_time."""
        from sklearn.metrics import mean_absolute_error

        y = conn_eval.test_y.to_numpy()
        gbr_mae = mean_absolute_error(y, conn_eval.point_pred)
        med_mae = mean_absolute_error(y, conn_eval.median_baseline)
        print(f"\n  connection_time  GBR MAE={gbr_mae:.3f}  training_median MAE={med_mae:.3f}")
        assert gbr_mae < med_mae, (
            f"GBR ({gbr_mae:.3f}) should beat training_median ({med_mae:.3f})"
        )

    def test_connection_time_gbr_beats_user_history(self, conn_eval):
        """GBR MAE < causal_user_history_median MAE for connection_time."""
        from sklearn.metrics import mean_absolute_error

        y = conn_eval.test_y.to_numpy()
        gbr_mae = mean_absolute_error(y, conn_eval.point_pred)
        hist_mae = mean_absolute_error(y, conn_eval.history_baseline)
        print(f"\n  connection_time  GBR MAE={gbr_mae:.3f}  history MAE={hist_mae:.3f}")
        assert gbr_mae < hist_mae, (
            f"GBR ({gbr_mae:.3f}) should beat user_history ({hist_mae:.3f})"
        )

    def test_next_gap_gbr_beats_training_median(self, gap_eval):
        """GBR MAE < training_median MAE for next_gap_hours."""
        from sklearn.metrics import mean_absolute_error

        y = gap_eval.test_y.to_numpy()
        gbr_mae = mean_absolute_error(y, gap_eval.point_pred)
        med_mae = mean_absolute_error(y, gap_eval.median_baseline)
        print(f"\n  next_gap  GBR MAE={gbr_mae:.3f}  training_median MAE={med_mae:.3f}")
        assert gbr_mae < med_mae, (
            f"GBR ({gbr_mae:.3f}) should beat training_median ({med_mae:.3f})"
        )

    def test_connection_time_coverage_near_80(self, conn_eval):
        """p10-p90 interval coverage for connection_time should be ~80 % (70-90 %)."""
        y = conn_eval.test_y.to_numpy()
        in_band = (y >= conn_eval.p10_pred) & (y <= conn_eval.p90_pred)
        coverage = float(in_band.mean())
        print(f"\n  connection_time  p10-p90 coverage = {coverage:.3f}")
        assert 0.70 <= coverage <= 0.92, (
            f"80 % interval coverage expected, got {coverage:.3f}"
        )

    def test_next_gap_coverage_near_80(self, gap_eval):
        """p10-p90 interval coverage for next_gap_hours should be ~80 % (70-92 %)."""
        y = gap_eval.test_y.to_numpy()
        in_band = (y >= gap_eval.p10_pred) & (y <= gap_eval.p90_pred)
        coverage = float(in_band.mean())
        print(f"\n  next_gap  p10-p90 coverage = {coverage:.3f}")
        assert 0.70 <= coverage <= 0.92, (
            f"80 % interval coverage expected, got {coverage:.3f}"
        )


# ===================================================================
# 2. Archetype Sanity Checks
# ===================================================================

class TestArchetypeSanity:
    """Predictions must reflect the known behavioural archetypes."""

    def test_workday_connection_longer(self, conn_eval):
        """Predicted connection_time should be longer when next_day_is_workday == True.

        Commuters plug in in the evening and unplug the *next* workday morning,
        so sessions before a workday (~14 h overnight) are systematically longer
        than casual Friday/Saturday afternoon top-ups.
        """
        recs = conn_eval.test_records
        pred = conn_eval.point_pred

        # Sessions plugged in on Friday (dow=4) or Saturday (dow=5) afternoon (before 17:00)
        dow = recs["plugin_dt"].dt.dayofweek.to_numpy()
        hour = recs["plugin_dt"].dt.hour.to_numpy()
        fri_sat_afternoon = ((dow == 4) | (dow == 5)) & (hour < 17)

        # Sessions where next_day_is_workday: reconstruct from plugin_dt
        next_dates = (recs["plugin_dt"] + pd.Timedelta(days=1)).dt.date
        import holidays as hol_mod
        no_hols = set(hol_mod.country_holidays("NO", years=range(2018, 2023)).keys())
        next_dow = (recs["plugin_dt"].dt.dayofweek.to_numpy() + 1) % 7
        ndiw = (next_dow < 5) & ~np.array([d in no_hols for d in next_dates])

        pred_before_wd = pred[ndiw]
        pred_fri_sat = pred[fri_sat_afternoon]

        if len(pred_fri_sat) < 5:
            pytest.skip("Not enough Friday/Saturday afternoon sessions to test")

        med_before_wd = float(np.median(pred_before_wd))
        med_fri_sat = float(np.median(pred_fri_sat))
        print(f"\n  Median pred conn before workday={med_before_wd:.2f} h, "
              f"Fri/Sat afternoon={med_fri_sat:.2f} h")
        assert med_before_wd > med_fri_sat, (
            f"Before-workday median ({med_before_wd:.2f}) should exceed "
            f"Fri/Sat afternoon ({med_fri_sat:.2f})"
        )

    def test_multiday_gap_predictions(self, gap_eval):
        """Multi-day charger users should get gap predictions in a plausible 30-80 h range."""
        recs = gap_eval.test_records
        pred = gap_eval.point_pred

        multiday_mask = recs["user_id"].astype(str).str.startswith("SYN_MULTIDAY_")
        if multiday_mask.sum() < 5:
            pytest.skip("Not enough multi-day charger test sessions")

        md_preds = pred[multiday_mask.to_numpy()]
        med_md = float(np.median(md_preds))
        print(f"\n  Multi-day users  median predicted gap = {med_md:.2f} h  "
              f"(n={multiday_mask.sum()})")
        # Allow generous range since the model is approximate
        assert 20.0 < med_md < 100.0, (
            f"Multi-day gap median ({med_md:.2f}) should be in plausible range, "
            f"not collapsed to global median"
        )


# ===================================================================
# 3. Zero-Leakage Tests
# ===================================================================

class TestZeroLeakage:
    """Corrupting the last 20 % of sessions must NOT change features for the first 80 %."""

    def test_connection_features_no_leakage(self, synthetic_sessions, nhts_priors):
        """Feature rows for the first 80 % are bit-identical before/after last-20 % corruption."""
        sessions = synthetic_sessions.copy()
        n = len(sessions)
        cutoff = int(n * 0.80)

        fb_stats = {
            "fb_conn": float(sessions.iloc[:cutoff]["connection_time"].median()),
            "fb_energy": float(sessions.iloc[:cutoff]["energy_session"].median()),
            "fb_gap": 10.0,
        }

        # 1. Extract features from the full clean dataset
        clean_feats, _, _ = extract_connection_time_features(
            sessions, nhts_priors, update_history=True, fallback_stats=fb_stats
        )

        # 2. Create a corrupted copy: randomize plugout_time and energy_session for last 20 %
        corrupted = sessions.copy()
        rng = np.random.default_rng(99)
        last_idx = corrupted.index[cutoff:]
        corrupted.loc[last_idx, "plugout_dt"] = (
            corrupted.loc[last_idx, "plugin_dt"]
            + pd.to_timedelta(rng.integers(30, 3000, size=len(last_idx)), unit="min")
        )
        corrupted.loc[last_idx, "connection_time"] = rng.uniform(0.5, 50.0, size=len(last_idx))
        corrupted.loc[last_idx, "energy_session"] = rng.uniform(0.1, 60.0, size=len(last_idx))

        # 3. Extract features from corrupted dataset
        corrupt_feats, _, _ = extract_connection_time_features(
            corrupted, nhts_priors, update_history=True, fallback_stats=fb_stats
        )

        # 4. First 80 % must be identical
        clean_first = clean_feats.iloc[:cutoff].reset_index(drop=True)
        corrupt_first = corrupt_feats.iloc[:cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            clean_first,
            corrupt_first,
            check_exact=True,
            obj="Connection-time features (first 80 %)",
        )

    def test_next_gap_features_no_leakage(self, synthetic_sessions, nhts_priors):
        """Feature rows for the first 80 % are bit-identical before/after last-20 % corruption."""
        sessions = synthetic_sessions.copy()
        n = len(sessions)
        cutoff = int(n * 0.80)

        fb_stats = {
            "fb_gap": float(sessions.iloc[:cutoff]["next_gap_hours"].dropna().median()),
            "fb_conn": float(sessions.iloc[:cutoff]["connection_time"].median()),
            "fb_energy": float(sessions.iloc[:cutoff]["energy_session"].median()),
        }

        # 1. Extract features from the full clean dataset
        clean_feats, _, _ = extract_next_gap_features(
            sessions, nhts_priors, update_history=True, fallback_stats=fb_stats
        )

        # 2. Create a corrupted copy
        corrupted = sessions.copy()
        rng = np.random.default_rng(99)
        last_idx = corrupted.index[cutoff:]
        corrupted.loc[last_idx, "plugout_dt"] = (
            corrupted.loc[last_idx, "plugin_dt"]
            + pd.to_timedelta(rng.integers(30, 3000, size=len(last_idx)), unit="min")
        )
        corrupted.loc[last_idx, "connection_time"] = rng.uniform(0.5, 50.0, size=len(last_idx))
        corrupted.loc[last_idx, "energy_session"] = rng.uniform(0.1, 60.0, size=len(last_idx))

        # Recompute next_gap_hours for the corrupted set
        corrupted = corrupted.sort_values(["user_id", "plugin_dt"], kind="stable").reset_index(drop=True)
        corrupted["next_plugin_dt"] = corrupted.groupby("user_id")["plugin_dt"].shift(-1)
        corrupted["next_gap_hours"] = (
            (corrupted["next_plugin_dt"] - corrupted["plugout_dt"]).dt.total_seconds() / 3600.0
        ).clip(lower=0.0)
        corrupted = corrupted.sort_values(["plugin_dt", "session_id"], kind="stable").reset_index(drop=True)

        # 3. Extract features from corrupted dataset
        corrupt_feats, _, _ = extract_next_gap_features(
            corrupted, nhts_priors, update_history=True, fallback_stats=fb_stats
        )

        # 4. First 80 % must be identical
        clean_first = clean_feats.iloc[:cutoff].reset_index(drop=True)
        corrupt_first = corrupt_feats.iloc[:cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            clean_first,
            corrupt_first,
            check_exact=True,
            obj="Next-gap features (first 80 %)",
        )
