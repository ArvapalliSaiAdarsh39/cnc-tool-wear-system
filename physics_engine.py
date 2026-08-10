"""
physics_engine.py
==================
First-principles machining physics used throughout the system.

v3 changes:
  * Taylor tool-life reparameterized onto the direct-exponent form
    T = C / (v^(1/n) * f^(1/m) * d^(1/p)) -- see config.py Section 1.
  * MRR reparameterized onto the axial/radial milling-center form
    MRR = ap * ae * vf / 1000 (cm^3/min), with vf (table feed, mm/min)
    computed from feed-per-rev * spindle RPM, plus multi-spindle scaling
    (MRR_total = N_spindles * MRR_single) for multi-spindle/multi-axis
    configurations.
  * Zero-safe by design: every function here treats a zero input (RPM,
    feed, or depth = 0) as a valid IDLE/STANDBY state and returns a sentinel
    result (cutting speed 0, tool life +inf, MRR 0) rather than raising --
    only genuinely invalid inputs (negative values) raise. This lets the
    dashboard evaluate the physics layer unconditionally on its zero
    defaults without a separate guard clause, though app.py still gates the
    full ML/SHAP/bulletin pipeline behind an explicit "is configured" check
    for UX reasons (showing a guided setup panel, not a wall of zeros).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config import (
    MaterialProfile,
    CoatingProfile,
    TAYLOR_LIFE_MIN_MIN,
    TAYLOR_LIFE_MAX_MIN,
    COOLANT_CONDITIONS,
    TOOL_COATINGS,
    SPINDLE_RATED_POWER_KW,
    SPINDLE_MECHANICAL_EFFICIENCY,
)


@dataclass(frozen=True)
class TaylorLifeResult:
    predicted_life_min: float   # clipped, "safe to use" value (or +inf at idle)
    raw_life_min: float         # unclipped raw power-law output
    is_extrapolated: bool
    is_idle: bool = False


@dataclass(frozen=True)
class ModifiedLifeResult:
    modified_life_min: float
    coolant_multiplier: float
    coating_multiplier: float
    is_compatible: bool
    compatibility_warning: str | None


class IncompatibleCoatingError(ValueError):
    """Raised when a coating is chemically incompatible with the selected material
    and the caller has not explicitly opted to proceed with a degraded estimate."""


def _require_nonnegative(**kwargs) -> None:
    for name, value in kwargs.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0 (got {value})")


def cutting_speed_m_min(rpm: float, diameter_mm: float) -> float:
    """
    Surface cutting speed Vc (m/min):   Vc = pi * D(mm) * N(rpm) / 1000

    RPM = 0 is a valid idle state (stationary spindle) and returns 0.0, not
    an error. A zero or negative diameter is not physically meaningful and
    still raises.
    """
    _require_nonnegative(rpm=rpm, diameter_mm=diameter_mm)
    if rpm == 0 or diameter_mm == 0:
        return 0.0
    return math.pi * diameter_mm * rpm / 1000.0


def taylor_tool_life(
    cutting_speed: float,
    feed_mm_rev: float,
    depth_mm: float,
    material: MaterialProfile,
) -> TaylorLifeResult:
    """
    Direct-exponent extended Taylor tool-life equation:

        T = C / (V^(1/n) * f^(1/m) * d^(1/p))

    where V = cutting speed (m/min), f = feed (mm/rev), d = axial depth of
    cut (mm), and C/n/m/p are the material constants in `material`.

    IDLE STATE: if V, f, or d is exactly 0 (the all-zero dashboard default,
    or any single zero input), tool life is undefined-but-infinite (no
    cutting is occurring, so nothing is wearing) -- this returns
    is_idle=True with predicted_life_min=float('inf') rather than raising a
    ZeroDivisionError. Negative inputs are still invalid and raise.

    Otherwise, the raw power-law result is clipped to
    [TAYLOR_LIFE_MIN_MIN, TAYLOR_LIFE_MAX_MIN] since power-law tool-life
    models are only valid inside their calibrated cutting-speed range;
    `is_extrapolated` reports whether clipping engaged.
    """
    _require_nonnegative(cutting_speed=cutting_speed, feed_mm_rev=feed_mm_rev, depth_mm=depth_mm)

    if cutting_speed == 0 or feed_mm_rev == 0 or depth_mm == 0:
        return TaylorLifeResult(predicted_life_min=float("inf"), raw_life_min=float("inf"),
                                 is_extrapolated=False, is_idle=True)

    denom = (cutting_speed ** (1.0 / material.n)) * (feed_mm_rev ** (1.0 / material.m)) * (depth_mm ** (1.0 / material.p))
    raw_life = material.C / denom

    clipped_life = min(max(raw_life, TAYLOR_LIFE_MIN_MIN), TAYLOR_LIFE_MAX_MIN)
    is_extrapolated = (raw_life < TAYLOR_LIFE_MIN_MIN) or (raw_life > TAYLOR_LIFE_MAX_MIN)

    return TaylorLifeResult(predicted_life_min=clipped_life, raw_life_min=raw_life,
                             is_extrapolated=is_extrapolated, is_idle=False)


def check_coating_compatibility(material: MaterialProfile, coating: CoatingProfile) -> tuple[bool, str | None]:
    """
    Returns (is_compatible, warning_message). PCD tooling reacts chemically
    with ferrous, nickel-based, AND titanium work materials at cutting
    temperatures (carbon diffuses into iron/nickel; titanium forms titanium
    carbide -- severe graphitization), causing rapid catastrophic edge
    failure rather than the gradual wear this system otherwise models.
    """
    if coating.incompatible_with_carbon_reactive and material.is_carbon_reactive:
        return False, (
            f"{coating.label} is chemically incompatible with {material.name} "
            f"(ferrous/nickel/titanium). Carbon diffusion or carbide formation at the cutting "
            f"interface causes rapid edge failure, not gradual wear -- select CBN or Ceramic "
            f"for this material instead."
        )
    return True, None


def apply_life_modifiers(
    base_life_min: float,
    material: MaterialProfile,
    coolant_key: str,
    coating_key: str,
    allow_incompatible: bool = False,
) -> ModifiedLifeResult:
    """Applies coolant-condition and tool-coating multipliers to a base
    Taylor life. Raises IncompatibleCoatingError for a chemically
    incompatible pairing unless allow_incompatible=True. Idle (infinite)
    base life passes through unchanged -- infinity * any positive
    multiplier is still infinity, which is the physically correct idle
    result."""
    coolant = COOLANT_CONDITIONS[coolant_key]
    coating = TOOL_COATINGS[coating_key]
    is_compatible, warning = check_coating_compatibility(material, coating)

    if not is_compatible:
        if not allow_incompatible:
            raise IncompatibleCoatingError(warning)
        coating_multiplier = 0.12
    else:
        coating_multiplier = coating.life_multiplier

    if math.isinf(base_life_min):
        modified_life = float("inf")
    else:
        modified_life = max(0.5, base_life_min * coolant.life_multiplier * coating_multiplier)

    return ModifiedLifeResult(
        modified_life_min=modified_life, coolant_multiplier=coolant.life_multiplier,
        coating_multiplier=coating_multiplier, is_compatible=is_compatible, compatibility_warning=warning,
    )


def table_feed_rate_mm_min(feed_mm_rev: float, rpm: float) -> float:
    """Table feed rate vf (mm/min) = feed per revolution (mm/rev) * spindle
    speed (RPM) -- the intermediate quantity the feed-per-tooth and MRR
    formulas are defined in terms of."""
    _require_nonnegative(feed_mm_rev=feed_mm_rev, rpm=rpm)
    return feed_mm_rev * rpm


def feed_per_tooth_mm(vf_mm_min: float, rpm: float, num_flutes: int) -> float:
    """
    Feed per tooth (mm):   fz = vf / (N * Z)

    vf = table feed rate (mm/min), N = spindle RPM, Z = number of
    simultaneously engaged cutting edges. Zero RPM is a valid idle state
    (returns 0.0, since vf is also 0 whenever RPM is 0).
    """
    if num_flutes < 1:
        raise ValueError("num_flutes must be >= 1")
    if rpm == 0:
        return 0.0
    return vf_mm_min / (rpm * num_flutes)


def material_removal_rate_cm3_min(ap_mm: float, ae_mm: float, vf_mm_min: float, num_spindles: int = 1) -> float:
    """
    Material Removal Rate, axial/radial (milling-center) form:

        MRR_single (cm^3/min) = ap * ae * vf / 1000

    ap = axial depth of cut (mm), ae = radial width of cut (mm),
    vf = table feed rate (mm/min). The /1000 converts mm^3/min to
    cm^3/min (1 cm^3 = 1000 mm^3).

    Multi-spindle / multi-axis scaling (volumetric removal is additive
    across simultaneously-cutting spindles doing identical work):

        MRR_total = N_spindles * MRR_single
    """
    if num_spindles < 1:
        raise ValueError("num_spindles must be >= 1")
    _require_nonnegative(ap_mm=ap_mm, ae_mm=ae_mm, vf_mm_min=vf_mm_min)
    mrr_single = (ap_mm * ae_mm * vf_mm_min) / 1000.0
    return mrr_single * num_spindles


def cutting_energy_kj(mrr_cm3_min: float, duration_min: float, specific_energy_j_mm3: float) -> float:
    """
    Cumulative cutting energy (kJ):
        Volume removed (mm^3) = MRR(cm^3/min) * 1000 * duration(min)
        Energy (kJ) = specific_energy(J/mm^3) * Volume(mm^3) / 1000
    """
    volume_mm3 = mrr_cm3_min * 1000.0 * duration_min
    return specific_energy_j_mm3 * volume_mm3 / 1000.0


def taylor_ratio(elapsed_runtime_min: float, taylor_predicted_life_min: float) -> float:
    """Taylor Ratio = elapsed_runtime / T_taylor. Returns 0.0 at idle
    (infinite life -> ratio is 0, not undefined) rather than NaN."""
    if math.isinf(taylor_predicted_life_min):
        return 0.0
    if taylor_predicted_life_min <= 0:
        return float("inf")
    return elapsed_runtime_min / taylor_predicted_life_min


def surface_roughness_ra_um(feed_rate_mm_rev: float, nose_radius_mm: float) -> float:
    """
    Theoretical (ideal) surface roughness for turning/milling (Ra, um):
        Ra ~= (f^2 / (32 * r_eps)) * 1000
    Zero feed (idle) is a valid input and returns Ra = 0.0.
    """
    _require_nonnegative(feed_rate_mm_rev=feed_rate_mm_rev, nose_radius_mm=nose_radius_mm)
    if nose_radius_mm == 0:
        return 0.0
    return (feed_rate_mm_rev ** 2 / (32.0 * nose_radius_mm)) * 1000.0


def cutting_power_kw(mrr_cm3_min: float, specific_energy_j_mm3: float) -> float:
    """
    Net cutting power (kW):
        P(kW) = specific_energy(J/mm^3) * MRR(cm^3/min) * 1000 / 60,000
              = specific_energy * MRR_cm3 / 60

    (MRR converted cm^3->mm^3 via *1000; J/min -> W via /60; W -> kW via /1000;
    the two /1000 and one *1000 combine to a single /60.)
    """
    return specific_energy_j_mm3 * mrr_cm3_min / 60.0


def spindle_torque_nm(power_kw: float, rpm: float) -> float:
    """Spindle torque (N*m): tau = P(W) / omega(rad/s). Zero RPM returns 0.0
    (no rotation, torque is not meaningfully defined as a divide-by-zero)."""
    if rpm == 0:
        return 0.0
    omega = 2.0 * math.pi * rpm / 60.0
    return (power_kw * 1000.0) / omega


def spindle_load_pct(
    cutting_power_kw_value: float,
    rated_power_kw: float = SPINDLE_RATED_POWER_KW,
    mechanical_efficiency: float = SPINDLE_MECHANICAL_EFFICIENCY,
) -> float:
    """Spindle load / power-utilization proxy (%): Load% = P_cut / (P_rated * eff) * 100."""
    available_power = rated_power_kw * mechanical_efficiency
    if available_power <= 0:
        return 0.0
    return min(200.0, (cutting_power_kw_value / available_power) * 100.0)
