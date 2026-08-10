"""
train_pipeline.py
==================
Offline/batch training entrypoint (CLI). This is deliberately separate from
app.py: in a real deployment, model (re)training is a scheduled batch job
(e.g. nightly, or triggered when enough new tool-life trials have been
logged) run by an MLOps pipeline, while the Streamlit dashboard is an online
serving surface that loads whatever artifact this job last produced. Keeping
them as separate processes is standard MLOps practice and lets training run
on a different machine/cadence than the dashboard.

Usage
-----
    python train_pipeline.py --n-runs 1500 --backend xgboost
    python train_pipeline.py --n-runs 2000 --backend lightgbm --seed 7

Produces:
    artifacts/wear_model.joblib
    artifacts/scrap_model.joblib
    artifacts/training_metrics.json
    data/synthetic_telemetry.csv
    data/featured_telemetry.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import config
from data_generator import generate_dataset
from feature_engineering import build_modeling_table
from ml_models import train_full_pipeline

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CNC Tool Wear quantile regressor and scrap-risk classifier."
    )
    parser.add_argument("--n-runs", type=int, default=config.N_SYNTHETIC_RUNS_DEFAULT,
                         help="Number of synthetic tool runs to simulate.")
    parser.add_argument("--backend", choices=["xgboost", "lightgbm"], default="xgboost",
                         help="Gradient-boosting backend for both the quantile and classification models.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed.")
    parser.add_argument("--target-coverage", type=float, default=0.80,
                         help="Target marginal coverage for the conformalized Q10-Q90 wear band.")
    parser.add_argument("--output-dir", default=".", help="Root directory for data/ and artifacts/ subfolders.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    data_dir = os.path.join(args.output_dir, config.DATA_DIR)
    model_dir = os.path.join(args.output_dir, config.MODEL_DIR)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    t0 = time.perf_counter()

    logger.info("Step 1/4 - Generating %d synthetic tool runs (seed=%d)...", args.n_runs, args.seed)
    raw = generate_dataset(n_runs=args.n_runs, seed=args.seed)
    raw.to_csv(os.path.join(data_dir, "synthetic_telemetry.csv"), index=False)
    logger.info("  -> %d raw telemetry rows across %d runs", len(raw), raw["run_id"].nunique())

    logger.info("Step 2/4 - Running feature engineering pipeline...")
    modeling_table = build_modeling_table(raw)
    modeling_table.to_csv(os.path.join(data_dir, "featured_telemetry.csv"), index=False)
    logger.info("  -> %d feature columns, %d modeling rows", len(config.FEATURE_COLUMNS), len(modeling_table))

    logger.info("Step 3/4 - Training quantile wear model + calibrated scrap classifier (backend=%s)...", args.backend)
    models, metrics = train_full_pipeline(
        modeling_table, backend=args.backend, target_coverage=args.target_coverage, random_state=args.seed
    )

    logger.info("Step 4/4 - Persisting model artifacts to %s ...", model_dir)
    models["wear_model"].save(os.path.join(model_dir, "wear_model.joblib"))
    models["scrap_model"].save(os.path.join(model_dir, "scrap_model.joblib"))
    with open(os.path.join(model_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Backend:                {args.backend}")
    print(f"Synthetic runs:         {args.n_runs}  (train={metrics['n_train_runs']}, "
          f"calib={metrics['n_calibration_runs']}, test={metrics['n_test_runs']})")
    print(f"Elapsed:                {elapsed:.1f}s")
    print("-" * 72)
    wear_post = metrics["quantile_wear_model_post_conformal"]
    print(f"Wear model (post-conformal):")
    print(f"  Q10/Q90 empirical coverage: {wear_post['q10_q90_empirical_coverage']*100:.1f}%  "
          f"(target {args.target_coverage*100:.0f}%)")
    print(f"  Mean band width:           {wear_post['mean_band_width_mm']:.4f} mm")
    print(f"  Conformal correction:      +/-{metrics['conformal_correction_mm']:.4f} mm")
    scrap = metrics["scrap_risk_classifier"]
    print(f"Scrap risk classifier:")
    print(f"  ROC-AUC:      {scrap['roc_auc']:.4f}")
    print(f"  Brier score:  {scrap['brier_score']:.4f}  (lower is better; 0 = perfect)")
    print("-" * 72)
    print(f"Artifacts written to:   {model_dir}/")
    print(f"Datasets written to:    {data_dir}/")
    print("=" * 72)


if __name__ == "__main__":
    main(sys.argv[1:])
