"""
ml_models.py
============
The Machine Learning Core of the system:

  1. QuantileWearModel — three gradient-boosted regressors (Q10 / Q50 / Q90)
     predicting flank wear (VB, mm), trained on the pinball (quantile) loss
     so the system reports an *uncertainty band* around every wear estimate
     rather than a single falsely-precise number. Supports XGBoost's native
     `reg:quantileerror` objective and LightGBM's native `quantile`
     objective interchangeably via a `backend` switch.

  2. ScrapRiskClassifier — a gradient-boosted classifier for the binary
     scrap/defect-risk event, wrapped in `CalibratedClassifierCV` so the
     output is a genuinely calibrated probability (a 0.7 really means
     "roughly 70% of samples like this end up scrapped in validation"),
     not just a monotonic score. Calibration matters enormously here because
     the whole point of this signal is to be read as a probability by a
     human operator.

Both classes expose a scikit-learn-like fit/predict interface, plus
save/load via joblib so a trained pipeline can be persisted as a deployable
artifact rather than retrained from scratch on every process start.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    mean_pinball_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

import config

logger = logging.getLogger(__name__)

Backend = Literal["xgboost", "lightgbm"]


# ---------------------------------------------------------------------------
# Quantile Wear Model
# ---------------------------------------------------------------------------
@dataclass
class QuantileWearModel:
    """
    Simultaneous Q10 / Q50 / Q90 gradient-boosted quantile regressors for
    flank wear (mm). Q50 is the expected-value operating estimate; the
    (Q10, Q90) band is the decision-support uncertainty range shown to the
    operator, per the "no false precision" safety principle of this system.
    """

    backend: Backend = "xgboost"
    quantiles: tuple = config.QUANTILES
    n_estimators: int = 350
    max_depth: int = 5
    learning_rate: float = 0.05
    random_state: int = config.RANDOM_SEED

    models_: Dict[float, object] = field(default_factory=dict, init=False, repr=False)
    feature_names_: Optional[List[str]] = field(default=None, init=False, repr=False)
    conformal_correction_mm_: float = field(default=0.0, init=False, repr=False)

    def _build_estimator(self, quantile: float):
        if self.backend == "xgboost":
            return XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.backend == "lightgbm":
            return LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=self.random_state,
                verbosity=-1,
            )
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Use 'xgboost' or 'lightgbm'.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileWearModel":
        self.feature_names_ = list(X.columns)
        for q in self.quantiles:
            logger.info("Training %s quantile regressor for q=%.2f", self.backend, q)
            model = self._build_estimator(q)
            model.fit(X, y)
            self.models_[q] = model
        return self

    def conformalize(self, X_calib: pd.DataFrame, y_calib: pd.Series, target_coverage: float = 0.80) -> float:
        """
        Split-Conformal Quantile Regression (CQR) calibration (Romano, Patterson
        & Candes, 2019). Raw independently-trained pinball-loss quantile
        models have NO coverage guarantee -- the 10th/90th percentile models
        can easily under- or over-cover on new data, as our own evaluation
        showed (~69-73% empirical coverage against an 80% target before this
        step). CQR fixes that with a single additive correction:

          1. On a held-out calibration set, compute the conformity score for
             each sample:   E_i = max(Q10(x_i) - y_i,  y_i - Q90(x_i))
             (how far, if at all, the true value fell outside the raw band)
          2. Take the ceil((n+1)*target_coverage)/n empirical quantile of
             {E_i}, call it Q_hat.
          3. At prediction time, widen every interval by Q_hat on both sides:
             [Q10(x) - Q_hat, Q90(x) + Q_hat]

        This gives a *finite-sample, distribution-free marginal coverage
        guarantee* of >= target_coverage on exchangeable data -- a much
        stronger statistical property than "we trained on the 0.1/0.9
        pinball loss and hope it's calibrated". The correction is stored on
        the model and applied automatically inside predict().
        """
        if not self.models_:
            raise RuntimeError("QuantileWearModel must be fit before conformalize().")

        q10_col, q90_col = self.quantiles[0], self.quantiles[-1]
        q10_pred = self.models_[q10_col].predict(X_calib)
        q90_pred = self.models_[q90_col].predict(X_calib)

        conformity_scores = np.maximum(q10_pred - y_calib.values, y_calib.values - q90_pred)

        n = len(conformity_scores)
        level = min(1.0, np.ceil((n + 1) * target_coverage) / n)
        q_hat = float(np.quantile(conformity_scores, level))

        self.conformal_correction_mm_ = max(0.0, q_hat)
        logger.info("CQR calibration: additive correction = %.4f mm (target coverage %.0f%%)",
                    self.conformal_correction_mm_, target_coverage * 100)
        return self.conformal_correction_mm_

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with one column per quantile (e.g. 'q10','q50','q90').

        Two corrections are applied on top of the three raw models:
          1. Quantile-crossing fix: since each quantile is trained as an
             independent model, nothing mathematically prevents Q10 > Q50 for
             a given row. Predictions are sorted per-row to enforce
             Q10 <= Q50 <= Q90, standard practice in production quantile
             regression pipelines.
          2. Conformal widening: if conformalize() has been run, the outer
             band (Q10, Q90) is widened by the stored conformal_correction_mm_
             so the reported interval carries its statistical coverage
             guarantee rather than just the raw (often overconfident) model
             output. Q50 is left untouched -- only the uncertainty band moves.
        """
        if not self.models_:
            raise RuntimeError("QuantileWearModel must be fit before predict().")

        raw = np.column_stack([self.models_[q].predict(X) for q in self.quantiles])
        raw_sorted = np.sort(raw, axis=1)  # enforce Q10 <= Q50 <= Q90 per row

        if self.conformal_correction_mm_ > 0:
            raw_sorted[:, 0] -= self.conformal_correction_mm_   # widen Q10 downward
            raw_sorted[:, -1] += self.conformal_correction_mm_  # widen Q90 upward

        raw_sorted = np.sort(raw_sorted, axis=1)   # re-sort in case widening crossed Q50
        raw_sorted = np.clip(raw_sorted, 0.0, None)  # wear cannot be negative

        col_names = [f"q{int(q * 100)}" for q in self.quantiles]
        return pd.DataFrame(raw_sorted, columns=col_names, index=X.index)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        preds = self.predict(X)
        metrics = {}
        for q, col in zip(self.quantiles, preds.columns):
            metrics[f"pinball_loss_{col}"] = mean_pinball_loss(y, preds[col], alpha=q)
        # Coverage: fraction of true values actually falling inside [Q10, Q90]
        q10_col, q90_col = preds.columns[0], preds.columns[-1]
        coverage = float(np.mean((y.values >= preds[q10_col].values) & (y.values <= preds[q90_col].values)))
        metrics["q10_q90_empirical_coverage"] = coverage
        metrics["mean_band_width_mm"] = float(np.mean(preds[q90_col].values - preds[q10_col].values))
        return metrics

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "QuantileWearModel":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Calibrated Scrap Risk Classifier
# ---------------------------------------------------------------------------
@dataclass
class ScrapRiskClassifier:
    """
    Calibrated binary classifier for scrap / defect risk. A raw gradient-
    boosted classifier is wrapped in CalibratedClassifierCV (isotonic
    regression by default) so predict_proba() returns a genuinely calibrated
    probability rather than an uncalibrated confidence score -- essential
    for a number that will be read directly by shop-floor operators as
    "percent chance of scrap".
    """

    backend: Backend = "xgboost"
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic"
    cv_folds: int = 4
    random_state: int = config.RANDOM_SEED

    model_: Optional[CalibratedClassifierCV] = field(default=None, init=False, repr=False)
    feature_names_: Optional[List[str]] = field(default=None, init=False, repr=False)

    def _build_base_estimator(self):
        if self.backend == "xgboost":
            return XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.backend == "lightgbm":
            return LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=self.random_state,
                verbosity=-1,
            )
        else:
            raise ValueError(f"Unknown backend '{self.backend}'. Use 'xgboost' or 'lightgbm'.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ScrapRiskClassifier":
        self.feature_names_ = list(X.columns)
        base = self._build_base_estimator()
        self.model_ = CalibratedClassifierCV(base, method=self.calibration_method, cv=self.cv_folds)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("ScrapRiskClassifier must be fit before predict_proba().")
        return self.model_.predict_proba(X)[:, 1]

    @staticmethod
    def risk_tier(probability: float) -> str:
        low, medium, high = config.RISK_TIER_EDGES
        if probability < low:
            return "Low"
        elif probability < medium:
            return "Medium"
        elif probability < high:
            return "High"
        return "Critical"

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        proba = self.predict_proba(X)
        return {
            "roc_auc": float(roc_auc_score(y, proba)),
            "brier_score": float(brier_score_loss(y, proba)),
        }

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ScrapRiskClassifier":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Orchestration helper
# ---------------------------------------------------------------------------
def train_full_pipeline(
    modeling_table: pd.DataFrame,
    backend: Backend = "xgboost",
    test_size: float = 0.2,
    calibration_size: float = 0.15,
    target_coverage: float = 0.80,
    random_state: int = config.RANDOM_SEED,
    n_estimators: int = 350,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    cv_folds: int = 4,
):
    """
    Three-way split BY run_id (never by row), so no single tool's telemetry
    samples leak across train / calibration / test:

      * train       -> fits the quantile regressors and the scrap classifier
      * calibration -> used ONLY for CQR conformal calibration of the
                        wear-quantile band (never seen during model fitting)
      * test        -> final held-out evaluation of everything

    `n_estimators` / `max_depth` / `learning_rate` / `cv_folds` are exposed
    so callers can trade model complexity for fit speed (e.g. app.py's
    instant-boot path uses a much smaller n_estimators/cv_folds on a small
    synthetic corpus; train_pipeline.py's CLI defaults reproduce the
    original full-precision configuration). Conformal calibration still
    applies at any scale -- it just yields a wider (more conservative) band
    when calibrated on fewer samples, which is the statistically correct
    behavior, not a bug of running small.

    Returns (models, metrics).
    """
    run_ids = modeling_table["run_id"].unique()
    train_val_ids, test_ids = train_test_split(run_ids, test_size=test_size, random_state=random_state)
    train_ids, calib_ids = train_test_split(
        train_val_ids, test_size=calibration_size / (1 - test_size), random_state=random_state
    )

    train_df = modeling_table[modeling_table["run_id"].isin(train_ids)]
    calib_df = modeling_table[modeling_table["run_id"].isin(calib_ids)]
    test_df = modeling_table[modeling_table["run_id"].isin(test_ids)]

    X_train, X_calib, X_test = (
        train_df[config.FEATURE_COLUMNS],
        calib_df[config.FEATURE_COLUMNS],
        test_df[config.FEATURE_COLUMNS],
    )
    y_train_reg, y_calib_reg, y_test_reg = (
        train_df[config.REGRESSION_TARGET],
        calib_df[config.REGRESSION_TARGET],
        test_df[config.REGRESSION_TARGET],
    )
    y_train_clf, y_test_clf = train_df[config.CLASSIFICATION_TARGET], test_df[config.CLASSIFICATION_TARGET]

    wear_model = QuantileWearModel(
        backend=backend, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate
    ).fit(X_train, y_train_reg)
    pre_calibration_metrics = wear_model.evaluate(X_test, y_test_reg)
    wear_model.conformalize(X_calib, y_calib_reg, target_coverage=target_coverage)
    post_calibration_metrics = wear_model.evaluate(X_test, y_test_reg)

    scrap_model = ScrapRiskClassifier(
        backend=backend, n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, cv_folds=cv_folds,
    ).fit(X_train, y_train_clf)

    metrics = {
        "quantile_wear_model_pre_conformal": pre_calibration_metrics,
        "quantile_wear_model_post_conformal": post_calibration_metrics,
        "conformal_correction_mm": wear_model.conformal_correction_mm_,
        "scrap_risk_classifier": scrap_model.evaluate(X_test, y_test_clf),
        "n_train_samples": len(train_df),
        "n_calibration_samples": len(calib_df),
        "n_test_samples": len(test_df),
        "n_train_runs": len(train_ids),
        "n_calibration_runs": len(calib_ids),
        "n_test_runs": len(test_ids),
    }
    gc.collect()
    return {"wear_model": wear_model, "scrap_model": scrap_model}, metrics


if __name__ == "__main__":
    from data_generator import generate_dataset
    from feature_engineering import build_modeling_table

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw = generate_dataset(n_runs=config.N_SYNTHETIC_RUNS_DEFAULT)
    table = build_modeling_table(raw)
    models, metrics = train_full_pipeline(table, backend="xgboost")

    print("\n=== Evaluation Metrics ===")
    for section, values in metrics.items():
        if isinstance(values, dict):
            print(f"\n{section}:")
            for k, v in values.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{section}: {values}")

    print("\n=== LightGBM backend cross-check ===")
    _, lgbm_metrics = train_full_pipeline(table, backend="lightgbm")
    print("scrap_risk_classifier (lightgbm):", lgbm_metrics["scrap_risk_classifier"])
    print("post-conformal coverage (lightgbm):",
          lgbm_metrics["quantile_wear_model_post_conformal"]["q10_q90_empirical_coverage"])
