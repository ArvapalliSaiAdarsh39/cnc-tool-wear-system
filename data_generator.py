"""
data_generator.py
==================
Physics-informed synthetic data engine (v3: axial/radial MRR with
multi-spindle scaling, direct-exponent Taylor form, 11 renamed materials).

Simulates a population of CNC tool "runs" (fresh insert -> retirement) as
telemetry time series. Per-run behavior is unchanged in spirit from v2:
material/coolant/coating sampled per run (coating weighted toward common
choices, with a deliberately small fraction landing on a chemically
incompatible pairing so the ML layer has real "this fails fast" examples to
learn from), three-stage non-linear flank-wear progression, and a
multi-factor scrap-probability model. New in v3: axial depth (ap) and
radial width (ae) are sampled as separate parameters feeding the
axial/radial MRR formula, and each run gets a spindle count for
multi-spindle volumetric scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from physics_engine import (
    cutting_speed_m_min,
    taylor_tool_life,
    apply_life_modifiers,
    table_feed_rate_mm_min,
    material_removal_rate_cm3_min,
    cutting_energy_kj,
    taylor_ratio,
    feed_per_tooth_mm,
    surface_roughness_ra_um,
    cutting_power_kw,
    spindle_load_pct,
)

_COATING_WEIGHTS = {
    "Uncoated": 0.28, "TiAlN": 0.27, "AlTiN": 0.20, "Ceramic": 0.10, "CBN": 0.10, "PCD": 0.05,
}
_COOLANT_WEIGHTS = {"Flood": 0.40, "MQL": 0.20, "HPC": 0.25, "Dry": 0.15}

_COATING_KEYS = list(_COATING_WEIGHTS.keys())
_COATING_P = np.array(list(_COATING_WEIGHTS.values()))
_COOLANT_KEYS = list(_COOLANT_WEIGHTS.keys())
_COOLANT_P = np.array(list(_COOLANT_WEIGHTS.values()))


def _wear_model(t_min, effective_life_min, tool_instance_factor, instability_accum, rng):
    """Three-stage non-linear flank wear (VB, mm) -- break-in / steady-state
    / catastrophic. Uses the coolant+coating-EFFECTIVE life so a poor
    coating choice shows up as faster wear, not just a different ratio."""
    ratio = t_min / effective_life_min if effective_life_min > 0 else 0.0
    breakin = config.WEAR_BREAKIN_MAX_MM * (1.0 - np.exp(-t_min / config.WEAR_BREAKIN_TAU_MIN))
    steady = config.WEAR_STEADY_COEF_MM * ratio * tool_instance_factor
    onset_excess = max(0.0, ratio - config.WEAR_CATASTROPHIC_ONSET_RATIO)
    catastrophic = config.WEAR_CATASTROPHIC_COEF_MM * (onset_excess ** 3)
    instability_term = config.WEAR_INSTABILITY_COEF_MM * instability_accum
    noise = rng.normal(0.0, config.WEAR_NOISE_STD_MM)
    return float(max(0.0, breakin + steady + catastrophic + instability_term + noise))


def _scrap_probability(vb_mm, instability_recent_ema):
    z = (config.SCRAP_LOGISTIC_STEEPNESS * (vb_mm - config.SCRAP_VB_THRESHOLD_MM)
         + config.SCRAP_INSTABILITY_COEF * instability_recent_ema)
    return 1.0 / (1.0 + np.exp(-z))


def _simulate_single_run(run_id: int, material_name: str, rng: np.random.Generator) -> pd.DataFrame:
    material = config.MATERIALS[material_name]

    rpm_setpoint = rng.uniform(*material.rpm_range)
    feed_setpoint = rng.uniform(*material.feed_range_mm_rev)
    ap_setpoint = rng.uniform(*material.ap_range_mm)
    ae_setpoint = rng.uniform(*material.ae_range_mm)
    diameter = rng.uniform(*config.WORKPIECE_DIAMETER_RANGE_MM)
    num_flutes = int(rng.integers(1, 9))
    num_spindles = int(rng.integers(config.NUM_SPINDLES_RANGE[0], config.NUM_SPINDLES_RANGE[1] + 1))
    nose_radius = float(rng.uniform(0.2, 1.6))

    coolant_key = rng.choice(_COOLANT_KEYS, p=_COOLANT_P)
    coating_key = rng.choice(_COATING_KEYS, p=_COATING_P)

    vc_setpoint = cutting_speed_m_min(rpm_setpoint, diameter)
    base_taylor = taylor_tool_life(vc_setpoint, feed_setpoint, ap_setpoint, material)
    life_mod = apply_life_modifiers(base_taylor.predicted_life_min, material, coolant_key, coating_key, allow_incompatible=True)
    effective_life = life_mod.modified_life_min

    max_runtime = effective_life * rng.uniform(*config.RUNTIME_STRETCH_RANGE)
    n_steps = max(1, int(max_runtime / config.SAMPLE_INTERVAL_MIN))
    tool_instance_factor = max(0.5, rng.normal(1.0, config.TOOL_INSTANCE_WEAR_FACTOR_STD))

    rows = []
    cumulative_energy = 0.0
    instability_accum = 0.0
    instability_recent_ema = 0.0

    for step in range(n_steps):
        t_min = step * config.SAMPLE_INTERVAL_MIN
        is_instability_event = rng.random() < config.INSTABILITY_EVENT_PROB
        noise_mult = config.INSTABILITY_NOISE_MULTIPLIER if is_instability_event else 1.0

        rpm_actual = max(1.0, rpm_setpoint + rng.normal(0, config.RPM_NOISE_STD_FRAC * rpm_setpoint * noise_mult))
        feed_actual = max(0.01, feed_setpoint + rng.normal(0, config.FEED_NOISE_STD_FRAC * feed_setpoint * noise_mult))
        ap_actual = max(0.05, ap_setpoint + rng.normal(0, config.DEPTH_NOISE_STD_FRAC * ap_setpoint * noise_mult))
        ae_actual = max(0.05, ae_setpoint + rng.normal(0, config.DEPTH_NOISE_STD_FRAC * ae_setpoint * noise_mult))

        vc_actual = cutting_speed_m_min(rpm_actual, diameter)
        vf_actual = table_feed_rate_mm_min(feed_actual, rpm_actual)
        mrr = material_removal_rate_cm3_min(ap_actual, ae_actual, vf_actual, num_spindles=num_spindles)
        energy_step = cutting_energy_kj(mrr, config.SAMPLE_INTERVAL_MIN, material.specific_energy_j_mm3)
        cumulative_energy += energy_step

        power_kw = cutting_power_kw(mrr, material.specific_energy_j_mm3)
        load_pct = spindle_load_pct(power_kw)
        fz = feed_per_tooth_mm(vf_actual, rpm_actual, num_flutes)
        ra = surface_roughness_ra_um(feed_actual, nose_radius)

        if is_instability_event:
            instability_accum += 1.0
        instability_recent_ema = config.INSTABILITY_EMA_DECAY * instability_recent_ema + (1.0 if is_instability_event else 0.0)

        t_ratio = taylor_ratio(t_min, effective_life)
        vb = _wear_model(t_min, effective_life, tool_instance_factor, instability_accum, rng)
        scrap_prob = _scrap_probability(vb, instability_recent_ema)
        scrap_event = int(rng.random() < scrap_prob)

        rows.append({
            "run_id": run_id, "material": material_name, "material_code": config.MATERIAL_TO_CODE[material_name],
            "coolant": coolant_key, "coolant_code": config.COOLANT_TO_CODE[coolant_key],
            "coating": coating_key, "coating_code": config.COATING_TO_CODE[coating_key],
            "coating_compatible": life_mod.is_compatible, "sample_index": step,
            "spindle_rpm": rpm_actual, "feed_rate_mm_rev": feed_actual,
            "axial_depth_mm": ap_actual, "radial_width_mm": ae_actual,
            "workpiece_diameter_mm": diameter, "num_flutes": num_flutes, "num_spindles": num_spindles,
            "nose_radius_mm": nose_radius, "cutting_speed_m_min": vc_actual, "runtime_min": t_min,
            "taylor_predicted_life_min": effective_life, "taylor_base_life_min": base_taylor.predicted_life_min,
            "taylor_extrapolated": base_taylor.is_extrapolated, "taylor_ratio": t_ratio,
            "mrr_cm3_min": mrr, "cumulative_energy_kj": cumulative_energy,
            "feed_per_tooth_mm": fz, "surface_roughness_ra_um": ra,
            "cutting_power_kw": power_kw, "spindle_load_pct": load_pct,
            "instability_event": int(is_instability_event), "instability_accum": instability_accum,
            "instability_recent_ema": instability_recent_ema, "flank_wear_mm": vb, "scrap_event": scrap_event,
        })

    return pd.DataFrame(rows)


def generate_dataset(n_runs: int = config.N_SYNTHETIC_RUNS_DEFAULT, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Generate a full synthetic telemetry dataset spanning `n_runs` simulated
    tool runs, evenly split across all 11 configured materials."""
    rng = np.random.default_rng(seed)
    material_names = list(config.MATERIALS.keys())
    all_runs = [
        _simulate_single_run(run_id, material_names[run_id % len(material_names)], rng)
        for run_id in range(n_runs)
    ]
    return pd.concat(all_runs, ignore_index=True)


if __name__ == "__main__":
    df = generate_dataset(n_runs=300)
    print(df.shape)
    print(df.groupby("material").agg(
        runs=("run_id", "nunique"), avg_effective_life=("taylor_predicted_life_min", "mean"),
        scrap_rate=("scrap_event", "mean"),
    ).round(2))
    print("\ncoating distribution:")
    print(df.drop_duplicates("run_id")["coating"].value_counts(normalize=True).round(3))
    print("\nincompatible-pairing rate:", (~df.drop_duplicates("run_id")["coating_compatible"]).mean())
    print("num_spindles range:", df["num_spindles"].min(), "-", df["num_spindles"].max())
