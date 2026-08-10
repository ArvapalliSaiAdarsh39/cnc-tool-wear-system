"""
shap_engine.py
===============
Explainable AI (XAI) layer. Produces local (single-run) SHAP feature
attributions for:

  1. The Q50 (expected-value) wear regressor -- "what is driving this run's
     expected flank wear estimate."
  2. The calibrated scrap-risk classifier -- "what is driving this run's
     scrap probability."

Design notes
------------
* SHAP's TreeExplainer needs a raw tree-based estimator. The scrap
  classifier is wrapped in `CalibratedClassifierCV`, which internally holds
  `cv_folds` independently-fitted copies of the base XGBoost/LightGBM model
  (one per cross-validation fold). Rather than picking just one fold
  (arbitrary) or bypassing calibration, this module computes SHAP values
  against EVERY fold's underlying tree model and averages them -- a small
  ensemble-of-explanations that is both more robust and uses 100% of the
  trained model rather than discarding the calibration wrapper's structure.
* Wear-model explanations are computed against the Q50 model only. Q50 is
  the "expected value" estimator; explaining all three quantile models
  separately would triple the SHAP compute for a dashboard whose actual
  job is "tell the operator what's driving the central estimate," not
  produce three redundant attribution sets that mostly agree anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import shap

import config
from ml_models import QuantileWearModel, ScrapRiskClassifier

# Human-readable labels for the raw feature columns, used in bar charts and
# fed to the LLM narrative agent so it never has to guess what a column name means.
FEATURE_DISPLAY_NAMES = {
    "material_code": "Material",
    "coolant_code": "Coolant Condition",
    "coating_code": "Tool Coating",
    "spindle_rpm": "Spindle Speed (RPM)",
    "feed_rate_mm_rev": "Feed Rate (mm/rev)",
    "axial_depth_mm": "Axial Depth of Cut, ap (mm)",
    "radial_width_mm": "Radial Width of Cut, ae (mm)",
    "workpiece_diameter_mm": "Workpiece Diameter (mm)",
    "num_flutes": "Number of Flutes/Edges (Z)",
    "num_spindles": "Number of Spindles",
    "nose_radius_mm": "Tool Nose Radius (mm)",
    "cutting_speed_m_min": "Cutting Speed (m/min)",
    "runtime_min": "Elapsed Runtime (min)",
    "taylor_predicted_life_min": "Taylor-Predicted Tool Life (min)",
    "taylor_ratio": "Taylor Ratio (runtime / predicted life)",
    "mrr_cm3_min": "Material Removal Rate (cm3/min)",
    "cumulative_energy_kj": "Cumulative Cutting Energy (kJ)",
    "feed_per_tooth_mm": "Feed per Tooth (mm)",
    "surface_roughness_ra_um": "Surface Roughness Ra (um)",
    "spindle_load_pct": "Spindle Load (%)",
    "rolling_rpm_std": "RPM Instability (rolling std)",
    "rolling_feed_std": "Feed Rate Instability (rolling std)",
    "rolling_depth_std": "Depth of Cut Instability (rolling std)",
}


@dataclass
class FeatureAttribution:
    feature: str
    display_name: str
    shap_value: float
    feature_value: float

    @property
    def direction(self) -> str:
        return "increases" if self.shap_value > 0 else "decreases"


@dataclass
class SHAPExplanation:
    base_value: float
    predicted_value: float
    attributions: List[FeatureAttribution] = field(default_factory=list)

    def top(self, n: int = 5) -> List[FeatureAttribution]:
        return sorted(self.attributions, key=lambda a: abs(a.shap_value), reverse=True)[:n]

    def as_records(self, n: int = 5) -> List[dict]:
        """JSON/plot-ready records for the dashboard bar chart and the LLM payload."""
        return [
            {
                "feature": a.feature,
                "display_name": a.display_name,
                "shap_value": round(a.shap_value, 5),
                "feature_value": a.feature_value,
                "direction": a.direction,
            }
            for a in self.top(n)
        ]


def _row_to_attributions(shap_row: np.ndarray, feature_row: pd.Series) -> List[FeatureAttribution]:
    return [
        FeatureAttribution(
            feature=feat,
            display_name=FEATURE_DISPLAY_NAMES.get(feat, feat),
            shap_value=float(shap_row[i]),
            feature_value=float(feature_row[feat]),
        )
        for i, feat in enumerate(config.FEATURE_COLUMNS)
    ]


def explain_wear_prediction(wear_model: QuantileWearModel, X_row: pd.DataFrame) -> SHAPExplanation:
    """
    Local SHAP explanation of the Q50 (expected-value) wear estimate for a
    single machining-run observation (X_row must be a 1-row DataFrame with
    columns == config.FEATURE_COLUMNS).
    """
    if len(X_row) != 1:
        raise ValueError("explain_wear_prediction expects exactly one row.")

    q50 = wear_model.quantiles[len(wear_model.quantiles) // 2]
    model = wear_model.models_[q50]

    explainer = shap.TreeExplainer(model)
    shap_output = explainer(X_row)

    attributions = _row_to_attributions(shap_output.values[0], X_row.iloc[0])
    predicted_value = float(model.predict(X_row)[0])

    return SHAPExplanation(
        base_value=float(np.array(shap_output.base_values).reshape(-1)[0]),
        predicted_value=predicted_value,
        attributions=attributions,
    )


def explain_scrap_risk(scrap_model: ScrapRiskClassifier, X_row: pd.DataFrame) -> SHAPExplanation:
    """
    Local SHAP explanation of the scrap-risk probability for a single
    observation, averaged across every cross-validation fold's underlying
    tree model inside the CalibratedClassifierCV wrapper (see module
    docstring for rationale).
    """
    if len(X_row) != 1:
        raise ValueError("explain_scrap_risk expects exactly one row.")
    if scrap_model.model_ is None:
        raise RuntimeError("ScrapRiskClassifier must be fit before explaining.")

    fold_shap_values = []
    fold_base_values = []

    for calibrated_fold in scrap_model.model_.calibrated_classifiers_:
        base_estimator = calibrated_fold.estimator
        explainer = shap.TreeExplainer(base_estimator)
        shap_output = explainer(X_row)

        # Binary classifiers: SHAP may return shape (1, n_features) for the
        # positive-class log-odds output, or (1, n_features, 2) for some
        # library/version combinations. Normalize to the positive class.
        values = shap_output.values
        if values.ndim == 3:
            values = values[:, :, 1]
        fold_shap_values.append(values[0])

        base = np.array(shap_output.base_values).reshape(-1)
        fold_base_values.append(base[-1] if base.size > 1 else base[0])

    averaged_shap = np.mean(fold_shap_values, axis=0)
    averaged_base = float(np.mean(fold_base_values))

    attributions = _row_to_attributions(averaged_shap, X_row.iloc[0])
    predicted_probability = float(scrap_model.predict_proba(X_row)[0])

    return SHAPExplanation(
        base_value=averaged_base,
        predicted_value=predicted_probability,
        attributions=attributions,
    )


if __name__ == "__main__":
    from data_generator import generate_dataset
    from feature_engineering import build_modeling_table
    from ml_models import train_full_pipeline

    raw = generate_dataset(n_runs=400)
    table = build_modeling_table(raw)
    models, _ = train_full_pipeline(table)

    sample_row = table[config.FEATURE_COLUMNS].iloc[[-1]]

    wear_explanation = explain_wear_prediction(models["wear_model"], sample_row)
    print("=== Wear (Q50) SHAP explanation ===")
    print(f"base value: {wear_explanation.base_value:.4f}  predicted: {wear_explanation.predicted_value:.4f}")
    for rec in wear_explanation.as_records():
        print(rec)

    scrap_explanation = explain_scrap_risk(models["scrap_model"], sample_row)
    print("\n=== Scrap risk SHAP explanation ===")
    print(f"base value: {scrap_explanation.base_value:.4f}  predicted proba: {scrap_explanation.predicted_value:.4f}")
    for rec in scrap_explanation.as_records():
        print(rec)
