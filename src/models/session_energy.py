"""Leakage-aware prediction of delivered energy for reported EV charging sessions.

The target is observed ``energy_session`` in Norway Dataset 1. Features are
available at plug-in time. User-history features are constructed causally: a
session never uses its own or a later session's observed energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
TARGET = "energy_session"
NUMERIC_FEATURES = [
    "plugin_hour_sin",
    "plugin_hour_cos",
    "plugin_weekday",
    "plugin_month_sin",
    "plugin_month_cos",
    "plugin_weekend",
    "history_count",
    "history_mean_kwh",
    "history_median_kwh",
]
CATEGORICAL_FEATURES = ["location"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class EvaluationData:
    """Feature matrices and baseline predictions for one evaluation task."""

    train_x: pd.DataFrame
    train_y: pd.Series
    test_x: pd.DataFrame
    test_y: pd.Series
    test_users: pd.Series
    median_baseline: np.ndarray
    history_baseline: np.ndarray


def load_reported_sessions(raw_path: Path) -> pd.DataFrame:
    """Load valid Dataset 1 records without using post-plug-in fields as features."""
    sessions = pd.read_csv(
        raw_path,
        sep=";",
        decimal=",",
        dtype={"location": "string", "user_id": "string", "session_id": "string"},
        low_memory=False,
    )
    sessions["plugin_dt"] = pd.to_datetime(sessions["plugin_time"], errors="coerce")
    sessions[TARGET] = pd.to_numeric(sessions[TARGET], errors="coerce")
    sessions = sessions.loc[
        sessions["plugin_dt"].notna() & sessions[TARGET].notna() & sessions[TARGET].ge(0)
    ].copy()
    if sessions.empty or sessions["session_id"].duplicated().any():
        raise ValueError("Expected nonempty records with unique session IDs")
    return sessions.sort_values(["plugin_dt", "session_id"], kind="stable").reset_index(drop=True)


def _time_features(records: pd.DataFrame) -> pd.DataFrame:
    clock_hour = records["plugin_dt"].dt.hour + records["plugin_dt"].dt.minute / 60
    month_phase = 2 * np.pi * (records["plugin_dt"].dt.month - 1) / 12
    return pd.DataFrame(
        {
            "plugin_hour_sin": np.sin(2 * np.pi * clock_hour / 24),
            "plugin_hour_cos": np.cos(2 * np.pi * clock_hour / 24),
            "plugin_weekday": records["plugin_dt"].dt.dayofweek.astype(float),
            "plugin_month_sin": np.sin(month_phase),
            "plugin_month_cos": np.cos(month_phase),
            "plugin_weekend": records["plugin_dt"].dt.dayofweek.ge(5).astype(float),
            "location": records["location"].astype("string"),
        },
        index=records.index,
    )


def causal_features(records: pd.DataFrame, initial_history: dict[str, list[float]] | None = None,
                    update_history: bool = True) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Build plug-in-time features using strictly earlier timestamps per user.

    All sessions sharing a timestamp receive features before any of their targets
    update the history. This avoids leakage between simultaneous records.
    """
    ordered = records.sort_values(["plugin_dt", "session_id"], kind="stable").copy()
    histories = {user: list(values) for user, values in (initial_history or {}).items()}
    feature_rows: list[dict[str, float | str]] = []

    for _, at_time in ordered.groupby("plugin_dt", sort=False):
        time_frame = _time_features(at_time)
        for row_index, row in at_time.iterrows():
            history = histories.get(str(row.user_id), [])
            row_features = time_frame.loc[row_index].to_dict()
            row_features["history_count"] = float(len(history))
            row_features["history_mean_kwh"] = float(np.mean(history)) if history else 0.0
            row_features["history_median_kwh"] = float(np.median(history)) if history else 0.0
            feature_rows.append({"row_index": row_index, **row_features})
        if update_history:
            for row in at_time.itertuples(index=False):
                histories.setdefault(str(row.user_id), []).append(float(getattr(row, TARGET)))

    features = pd.DataFrame(feature_rows).set_index("row_index").reindex(ordered.index)
    return features[FEATURES], histories


def _history_from_records(records: pd.DataFrame) -> dict[str, list[float]]:
    """Return per-user observations ordered through the end of a training interval."""
    history: dict[str, list[float]] = {}
    for row in records.sort_values(["plugin_dt", "session_id"], kind="stable").itertuples(index=False):
        history.setdefault(str(row.user_id), []).append(float(getattr(row, TARGET)))
    return history


def make_evaluation(train: pd.DataFrame, test: pd.DataFrame,
                    allow_test_history: bool) -> EvaluationData:
    """Prepare train/test matrices and causal baselines.

    ``allow_test_history`` is true only for the future-time task. It permits a
    later test session to use prior observed test-session energy, as would be
    available after each session completes in an online deployment. It is false
    for the unseen-user task, where test users are absent from training.
    """
    train_x, _ = causal_features(train)
    history = _history_from_records(train)
    test_x, _ = causal_features(test, initial_history=history, update_history=allow_test_history)
    train_y = train[TARGET].copy()
    test_y = test[TARGET].copy()
    training_median = float(train_y.median())
    history_baseline = np.where(
        test_x["history_count"].to_numpy() > 0,
        test_x["history_median_kwh"].to_numpy(),
        training_median,
    )
    return EvaluationData(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        test_users=test["user_id"].astype("string"),
        median_baseline=np.full(len(test), training_median),
        history_baseline=history_baseline,
    )


def build_model() -> Pipeline:
    """Return one efficient tabular gradient-boosting regression pipeline."""
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("location", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("features", preprocessor),
            ("model", HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=1.0, random_state=RANDOM_SEED,
            )),
        ]
    )


def _metric_rows(evaluation_name: str, data: EvaluationData,
                 predictions: dict[str, np.ndarray], subsets: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subset_name, mask in subsets.items():
        for model_name, predicted in predictions.items():
            rows.append(
                {
                    "evaluation": evaluation_name,
                    "subset": subset_name,
                    "predictor": model_name,
                    "n_sessions": int(mask.sum()),
                    "n_users": int(data.test_users[mask].nunique()),
                    "mae_kwh": mean_absolute_error(data.test_y[mask], predicted[mask]),
                    "rmse_kwh": root_mean_squared_error(data.test_y[mask], predicted[mask]),
                }
            )
    return rows


def run_experiment(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run both required evaluations and write requested output artifacts."""
    sessions = load_reported_sessions(root / "data" / "raw" / "norway_ev" / "Dataset1_charging_reports.csv")
    cutoff_index = int(len(sessions) * 0.8)
    cutoff = sessions.loc[cutoff_index, "plugin_dt"]
    temporal_train = sessions.loc[sessions.plugin_dt < cutoff].copy()
    temporal_test = sessions.loc[sessions.plugin_dt >= cutoff].copy()
    temporal = make_evaluation(temporal_train, temporal_test, allow_test_history=True)
    temporal_model = build_model().fit(temporal.train_x, temporal.train_y)
    temporal_prediction = temporal_model.predict(temporal.test_x)
    seen_users = set(temporal_train.user_id.astype(str))
    warm_mask = temporal.test_users.astype(str).isin(seen_users).to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    group_train_idx, group_test_idx = next(splitter.split(sessions, groups=sessions.user_id))
    group_train = sessions.iloc[group_train_idx].copy()
    group_test = sessions.iloc[group_test_idx].copy()
    grouped = make_evaluation(group_train, group_test, allow_test_history=False)
    grouped_model = build_model().fit(grouped.train_x, grouped.train_y)
    grouped_prediction = grouped_model.predict(grouped.test_x)

    metrics = pd.DataFrame(
        _metric_rows(
            "future_session_chronological",
            temporal,
            {
                "training_median": temporal.median_baseline,
                "causal_user_history_median": temporal.history_baseline,
                "gradient_boosting": temporal_prediction,
            },
            {"all_test": np.ones(len(temporal.test_y), dtype=bool), "warm_user": warm_mask,
             "cold_start_user": ~warm_mask},
        )
        + _metric_rows(
            "unseen_user_group_holdout",
            grouped,
            {
                "training_median": grouped.median_baseline,
                "causal_user_history_median": grouped.history_baseline,
                "gradient_boosting": grouped_prediction,
            },
            {"all_test": np.ones(len(grouped.test_y), dtype=bool)},
        )
    )

    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)
    metrics.to_csv(outputs / "ml_metrics.csv", index=False, float_format="%.6f")
    prediction_table = pd.DataFrame(
        {
            "plugin_dt": temporal_test.plugin_dt.to_numpy(),
            "user_id": temporal_test.user_id.to_numpy(),
            "actual_energy_kwh": temporal.test_y.to_numpy(),
            "predicted_energy_kwh": temporal_prediction,
            "user_seen_in_training": warm_mask,
        }
    )
    _save_actual_vs_predicted(prediction_table, outputs / "ml_actual_vs_predicted.png")

    metadata = {
        "target": TARGET,
        "temporal_cutoff": str(cutoff),
        "temporal_train_sessions": len(temporal_train),
        "temporal_test_sessions": len(temporal_test),
        "temporal_train_users": temporal_train.user_id.nunique(),
        "temporal_test_users": temporal_test.user_id.nunique(),
        "temporal_warm_sessions": int(warm_mask.sum()),
        "temporal_cold_sessions": int((~warm_mask).sum()),
        "group_train_sessions": len(group_train),
        "group_test_sessions": len(group_test),
        "group_train_users": group_train.user_id.nunique(),
        "group_test_users": group_test.user_id.nunique(),
    }
    return metrics, prediction_table, metadata


def _save_actual_vs_predicted(predictions: pd.DataFrame, output_path: Path) -> None:
    """Save a readable future-session held-out actual-vs-predicted figure."""
    plot_data = predictions.sample(min(3_000, len(predictions)), random_state=RANDOM_SEED)
    limit = max(50, float(np.quantile(np.r_[plot_data.actual_energy_kwh, plot_data.predicted_energy_kwh], 0.99)))
    fig, ax = plt.subplots(figsize=(7.3, 6.0))
    ax.scatter(
        plot_data.actual_energy_kwh,
        plot_data.predicted_energy_kwh,
        s=12,
        alpha=0.30,
        color="#2463a6",
        edgecolors="none",
        label="Held-out future sessions",
    )
    ax.plot([0, limit], [0, limit], color="#d97732", linewidth=1.5, label="Perfect prediction")
    ax.set(
        xlim=(0, limit), ylim=(0, limit), aspect="equal",
        xlabel="Observed delivered energy (kWh)", ylabel="Predicted delivered energy (kWh)",
        title="Future-session energy prediction: chronological test set",
    )
    ax.grid(color="#e1e6eb", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    result_metrics, _, result_metadata = run_experiment(project_root)
    print(pd.DataFrame([result_metadata]).T.rename(columns={0: "value"}).to_string())
    print(result_metrics.to_string(index=False))
