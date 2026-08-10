"""
feature_engineering.py
=======================
Takes the raw physics-informed telemetry stream from data_generator.py and
adds/finalizes the modeling table.

v2 additions:
  * The raw stream now already carries the expanded physics fields
    (feed-per-tooth, surface roughness, cutting power, spindle load,
    coolant/coating codes) computed inline in data_generator.py -- this
    module's job is unchanged in spirit: rolling-window instability
    features, then assemble the final modeling table matching
    config.FEATURE_COLUMNS.
  * `fit_feature_scaler` / `apply_feature_scaler`: an optional standardization
    utility (scikit-learn StandardScaler) over the numeric feature columns.
    The shipped quantile/classification models are gradient-boosted trees,
    which are scale-invariant and do not require this -- it is provided so
    (a) any future non-tree model (linear/neural baselines, clustering,
    anomaly detection on the same feature table) has a ready-made,
    consistently-fit scaler rather than every consumer re-deriving one, and
    (b) numeric feature values can be reported on a comparable scale in any
    downstream analysis/reporting that wants it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config

# Categorical/code columns are intentionally excluded from scaling -- they
# are small-cardinality integer codes consumed natively by the tree models,
# not continuous quantities that benefit from standardization.
_CATEGORICAL_FEATURE_COLUMNS = {"material_code", "coolant_code", "coating_code", "num_flutes", "num_spindles"}
NUMERIC_SCALABLE_COLUMNS = [c for c in config.FEATURE_COLUMNS if c not in _CATEGORICAL_FEATURE_COLUMNS]


def add_rolling_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing rolling standard deviations per run_id, sorted by
    sample_index (config.ROLLING_WINDOW_SAMPLES window, min_periods=1 so
    early-life samples still get a noisier rolling estimate rather than NaN)."""
    df = df.sort_values(["run_id", "sample_index"]).copy()
    grouped = df.groupby("run_id", sort=False)

    for src_col, out_col in [
        ("spindle_rpm", "rolling_rpm_std"),
        ("feed_rate_mm_rev", "rolling_feed_std"),
        ("axial_depth_mm", "rolling_depth_std"),
    ]:
        df[out_col] = (
            grouped[src_col]
            .rolling(window=config.ROLLING_WINDOW_SAMPLES, min_periods=1)
            .std()
            .fillna(0.0)
            .reset_index(level=0, drop=True)
        )
    return df


def build_modeling_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Full feature-engineering pass -> clean modeling table containing
    exactly config.FEATURE_COLUMNS plus the two prediction targets and a
    few passthrough identifier/context columns useful for SHAP narratives,
    the PDF work-order export, and dashboard display.

    Numeric feature columns are cast to float32 (single precision) to
    roughly halve the in-memory footprint of the training table -- a real
    consideration on Streamlit Community Cloud's constrained memory tier.
    Targets stay float64 (xgboost/sklearn's internal loss computation
    expects it) and small-cardinality code columns stay as compact ints."""
    featured = add_rolling_variance_features(raw_df)

    passthrough_cols = [
        "run_id", "sample_index", "material", "coolant", "coating", "coating_compatible",
        "instability_event", "instability_accum", "instability_recent_ema",
        "taylor_extrapolated", "taylor_base_life_min", "cutting_power_kw",
    ]
    keep_cols = config.FEATURE_COLUMNS + [config.REGRESSION_TARGET, config.CLASSIFICATION_TARGET] + passthrough_cols

    seen = set()
    ordered_unique_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
    table = featured[ordered_unique_cols].reset_index(drop=True)

    for col in NUMERIC_SCALABLE_COLUMNS:
        table[col] = table[col].astype("float32")
    for col in _CATEGORICAL_FEATURE_COLUMNS:
        table[col] = table[col].astype("int32")

    return table


def engineer_features_for_single_observation(raw_observation: dict) -> dict:
    """Feature-engineering path for a SINGLE live observation (the Streamlit
    dashboard's 'what if I ran at these parameters' query). Rolling std
    defaults to 0.0 (perfectly steady) unless the caller supplies a value
    from a short recent-history buffer."""
    obs = dict(raw_observation)
    obs.setdefault("rolling_rpm_std", 0.0)
    obs.setdefault("rolling_feed_std", 0.0)
    obs.setdefault("rolling_depth_std", 0.0)
    return {col: obs[col] for col in config.FEATURE_COLUMNS}


@dataclass
class FittedScaler:
    scaler: StandardScaler
    columns: List[str]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out[self.columns] = self.scaler.transform(X[self.columns])
        return out

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out[self.columns] = self.scaler.inverse_transform(X[self.columns])
        return out

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FittedScaler":
        return joblib.load(path)


def fit_feature_scaler(X_train: pd.DataFrame, columns: Optional[List[str]] = None) -> FittedScaler:
    """Fit a StandardScaler over the numeric (non-categorical-code) feature
    columns of a training split. See module docstring for why this is
    optional/supplementary rather than applied to the shipped tree models."""
    cols = columns or NUMERIC_SCALABLE_COLUMNS
    scaler = StandardScaler()
    scaler.fit(X_train[cols])
    return FittedScaler(scaler=scaler, columns=cols)


def apply_feature_scaler(X: pd.DataFrame, fitted: FittedScaler) -> pd.DataFrame:
    return fitted.transform(X)


if __name__ == "__main__":
    from data_generator import generate_dataset

    raw = generate_dataset(n_runs=150)
    table = build_modeling_table(raw)
    print(table.shape)
    print(table[config.FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]])

    scaler = fit_feature_scaler(table[config.FEATURE_COLUMNS])
    scaled = apply_feature_scaler(table[config.FEATURE_COLUMNS], scaler)
    print("\nScaled numeric columns (mean ~0, std ~1):")
    print(scaled[NUMERIC_SCALABLE_COLUMNS].describe().T[["mean", "std"]].round(3))
    print("\nCategorical code columns left untouched:")
    print(scaled[list(_CATEGORICAL_FEATURE_COLUMNS)].describe().T[["mean", "std"]].round(3))
