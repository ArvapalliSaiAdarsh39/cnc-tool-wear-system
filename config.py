"""
config.py
=========
Single source of truth for the CNC Tool Wear Prediction & Scrap Risk Signal System.

v3 changes:
  * Materials renamed to full human-readable engineering names (11 total --
    Hastelloy C-276 and Ductile Iron dropped per the current spec's explicit
    whitelist) and reparameterized onto the direct-exponent Taylor form
    T = C / (v^(1/n) * f^(1/m) * d^(1/p)), replacing the v1/v2 grouped-exponent
    form. Coolant/coating multipliers are unchanged in spirit -- still applied
    on top of the base Taylor life.
  * New radial width of cut (ae) parameter and axial/radial MRR formula
    (MRR = ap * ae * vf / 1000, cm^3/min), replacing the turning-style
    MRR = 1000 * Vc * f * d (mm^3/min) used previously.
  * New spindle-count parameter for multi-spindle volumetric scaling.
  * Safe operating envelopes narrowed to realistic production windows
    (previously spanned rough-to-finish extremes); this matters more now
    because the new exponent ranges (particularly p, the depth exponent)
    make T extremely sensitive to parameter spread -- a narrow, realistic
    window keeps the physics-model output well-behaved without changing
    the specified exponent ranges themselves.

IMPORTANT ENGINEERING NOTE ON THE CONSTANTS BELOW
--------------------------------------------------
Taylor coefficients (C, n, m, p), specific cutting energy, hardness, and
machinability ratings are illustrative, literature-directional defaults --
correctly ORDERED and calibrated so the system behaves realistically, but
NOT certified metallurgical constants for any specific insert grade/lot.
Re-fit against real ISO 3685 tool-life trials before using this system's
numbers to drive real production decisions -- see docs/ARCHITECTURE.md.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# 1. MATERIAL PHYSICS CONSTANTS (Direct-Exponent Extended Taylor Model)
# ---------------------------------------------------------------------------
# T = C / (v^(1/n) * f^(1/m) * d^(1/p))
#   v = cutting speed (m/min), f = feed (mm/rev), d = axial depth of cut (mm)
#   n in [0.20, 0.50] -- speed sensitivity (higher n = less speed-sensitive)
#   m in [0.15, 0.30] -- feed sensitivity
#   p in [0.10, 0.15] -- depth sensitivity
# machinability_pct: relative-to-reference machinability index, display only.

@dataclass(frozen=True)
class MaterialProfile:
    name: str                    # full human-readable engineering name
    short_code: str              # standard designation, e.g. "AISI 4140"
    family: str
    C: float
    n: float
    m: float
    p: float
    specific_energy_j_mm3: float
    rpm_range: Tuple[float, float]
    feed_range_mm_rev: Tuple[float, float]
    ap_range_mm: Tuple[float, float]   # axial depth of cut
    ae_range_mm: Tuple[float, float]   # radial width of cut
    hardness_hb: Tuple[float, float]
    machinability_pct: float
    is_carbon_reactive: bool     # ferrous, nickel-based, OR titanium -- drives the Diamond/PCD guard


MATERIALS: Dict[str, MaterialProfile] = {
    "Aluminium Alloy (Al 6061-T6)": MaterialProfile(
        "Aluminium Alloy (Al 6061-T6)", "Al 6061-T6", "Aluminium Alloys",
        C=5.689e6, n=0.50, m=0.30, p=0.150, specific_energy_j_mm3=0.75,
        rpm_range=(2600, 3800), feed_range_mm_rev=(0.20, 0.34), ap_range_mm=(1.2, 2.2), ae_range_mm=(1.5, 3.0),
        hardness_hb=(90, 100), machinability_pct=90, is_carbon_reactive=False,
    ),
    "Aluminium Alloy (Al 7075-T6)": MaterialProfile(
        "Aluminium Alloy (Al 7075-T6)", "Al 7075-T6", "Aluminium Alloys",
        C=3.589e6, n=0.46, m=0.28, p=0.145, specific_energy_j_mm3=0.95,
        rpm_range=(2200, 3300), feed_range_mm_rev=(0.18, 0.30), ap_range_mm=(1.1, 2.0), ae_range_mm=(1.4, 2.8),
        hardness_hb=(140, 160), machinability_pct=70, is_carbon_reactive=False,
    ),
    "Medium Carbon Steel (AISI 1045)": MaterialProfile(
        "Medium Carbon Steel (AISI 1045)", "AISI 1045", "Carbon & Tool Steels",
        C=2.821e6, n=0.36, m=0.24, p=0.135, specific_energy_j_mm3=2.2,
        rpm_range=(1000, 1600), feed_range_mm_rev=(0.16, 0.28), ap_range_mm=(1.0, 1.8), ae_range_mm=(1.2, 2.2),
        hardness_hb=(170, 210), machinability_pct=57, is_carbon_reactive=True,
    ),
    "Alloy Steel (AISI 4140 / 42CrMo4)": MaterialProfile(
        "Alloy Steel (AISI 4140 / 42CrMo4)", "AISI 4140", "Carbon & Tool Steels",
        C=1.396e6, n=0.32, m=0.22, p=0.130, specific_energy_j_mm3=2.5,
        rpm_range=(850, 1350), feed_range_mm_rev=(0.14, 0.24), ap_range_mm=(0.9, 1.6), ae_range_mm=(1.1, 2.0),
        hardness_hb=(197, 247), machinability_pct=52, is_carbon_reactive=True,
    ),
    "High-Carbon Tool Steel (AISI D2)": MaterialProfile(
        "High-Carbon Tool Steel (AISI D2)", "AISI D2", "Carbon & Tool Steels",
        C=7776.0, n=0.28, m=0.20, p=0.125, specific_energy_j_mm3=3.1,
        rpm_range=(450, 750), feed_range_mm_rev=(0.10, 0.18), ap_range_mm=(0.6, 1.1), ae_range_mm=(0.7, 1.4),
        hardness_hb=(230, 260), machinability_pct=38, is_carbon_reactive=True,
    ),
    "Austenitic Stainless Steel (AISI 304)": MaterialProfile(
        "Austenitic Stainless Steel (AISI 304)", "AISI 304", "Stainless Steels",
        C=1.912e5, n=0.27, m=0.19, p=0.122, specific_energy_j_mm3=2.9,
        rpm_range=(600, 950), feed_range_mm_rev=(0.12, 0.21), ap_range_mm=(0.7, 1.3), ae_range_mm=(0.9, 1.7),
        hardness_hb=(150, 200), machinability_pct=45, is_carbon_reactive=True,
    ),
    "Austenitic Stainless Steel (AISI 316L)": MaterialProfile(
        "Austenitic Stainless Steel (AISI 316L)", "AISI 316L", "Stainless Steels",
        C=1.602e5, n=0.25, m=0.18, p=0.118, specific_energy_j_mm3=3.1,
        rpm_range=(550, 880), feed_range_mm_rev=(0.11, 0.20), ap_range_mm=(0.7, 1.2), ae_range_mm=(0.85, 1.6),
        hardness_hb=(150, 200), machinability_pct=40, is_carbon_reactive=True,
    ),
    "Precipitation Hardening Stainless Steel (17-4 PH)": MaterialProfile(
        "Precipitation Hardening Stainless Steel (17-4 PH)", "17-4 PH", "Stainless Steels",
        C=6840.0, n=0.24, m=0.17, p=0.115, specific_energy_j_mm3=3.4,
        rpm_range=(470, 740), feed_range_mm_rev=(0.10, 0.17), ap_range_mm=(0.55, 1.0), ae_range_mm=(0.7, 1.3),
        hardness_hb=(375, 420), machinability_pct=32, is_carbon_reactive=True,
    ),
    "Titanium Alloy (Ti-6Al-4V)": MaterialProfile(
        "Titanium Alloy (Ti-6Al-4V)", "Ti-6Al-4V", "Superalloys & Titanium",
        C=140.1, n=0.22, m=0.16, p=0.112, specific_energy_j_mm3=3.8,
        rpm_range=(220, 380), feed_range_mm_rev=(0.08, 0.15), ap_range_mm=(0.5, 0.95), ae_range_mm=(0.6, 1.2),
        hardness_hb=(330, 350), machinability_pct=25, is_carbon_reactive=True,
    ),
    "Nickel Superalloy (Inconel 718)": MaterialProfile(
        "Nickel Superalloy (Inconel 718)", "Inconel 718", "Superalloys & Titanium",
        C=0.1244, n=0.20, m=0.15, p=0.100, specific_energy_j_mm3=5.4,
        rpm_range=(160, 270), feed_range_mm_rev=(0.06, 0.11), ap_range_mm=(0.35, 0.65), ae_range_mm=(0.45, 0.85),
        hardness_hb=(350, 450), machinability_pct=13, is_carbon_reactive=True,
    ),
    "Gray Cast Iron (FG 15 / CGI 450)": MaterialProfile(
        "Gray Cast Iron (FG 15 / CGI 450)", "Gray Cast Iron", "Cast Irons",
        C=3.105e6, n=0.40, m=0.26, p=0.140, specific_energy_j_mm3=1.4,
        rpm_range=(750, 1150), feed_range_mm_rev=(0.19, 0.32), ap_range_mm=(1.2, 2.2), ae_range_mm=(1.5, 2.8),
        hardness_hb=(150, 220), machinability_pct=70, is_carbon_reactive=True,
    ),
}

WORKPIECE_DIAMETER_RANGE_MM: Tuple[float, float] = (10.0, 120.0)
NUM_SPINDLES_RANGE: Tuple[int, int] = (1, 6)

# ---------------------------------------------------------------------------
# 2. COOLANT CONDITION -- tool-life multiplier
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoolantProfile:
    label: str
    life_multiplier: float
    description: str


COOLANT_CONDITIONS: Dict[str, CoolantProfile] = {
    "Dry": CoolantProfile(
        "Dry Machining", 0.55,
        "No cutting fluid. Highest thermal stress at the cutting edge; shortens tool life "
        "substantially unless the coating/material pairing is specifically rated for dry cutting.",
    ),
    "MQL": CoolantProfile(
        "Minimum Quantity Lubrication (MQL)", 0.80,
        "A fine mist of lubricant at low flow. Reduces friction and some heat, but far less "
        "thermal mass than flood coolant -- a middle ground on tool life.",
    ),
    "Flood": CoolantProfile(
        "Flood Coolant", 1.00,
        "Conventional high-volume flood cooling. Reference/baseline condition for the Taylor "
        "life constants shipped in this system.",
    ),
    "HPC": CoolantProfile(
        "High-Pressure Coolant (HPC)", 1.25,
        "Coolant delivered at high pressure directly at the cutting zone, improving chip "
        "evacuation and heat removal -- extends tool life beyond flood, especially in gummy "
        "or deep cuts, at the cost of equipment complexity.",
    ),
}

# ---------------------------------------------------------------------------
# 3. TOOL COATING / SUBSTRATE -- tool-life multiplier + compatibility
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CoatingProfile:
    label: str
    life_multiplier: float
    incompatible_with_carbon_reactive: bool
    description: str


TOOL_COATINGS: Dict[str, CoatingProfile] = {
    "Uncoated": CoatingProfile(
        "Uncoated Cemented Carbide (Grade K/P)", 1.00, False,
        "Baseline tungsten-carbide substrate, no coating. Reference condition for the Taylor "
        "life constants shipped in this system.",
    ),
    "TiAlN": CoatingProfile(
        "TiAlN-Coated Carbide", 1.80, False,
        "Titanium aluminum nitride PVD coating. Good hot hardness and oxidation resistance; a "
        "common general-purpose upgrade over uncoated carbide.",
    ),
    "AlTiN": CoatingProfile(
        "AlTiN-Coated Carbide", 2.00, False,
        "Aluminum-rich variant of TiAlN with even higher hot hardness -- favored for "
        "high-speed and dry/near-dry machining of steels and superalloys.",
    ),
    "PCD": CoatingProfile(
        "Polycrystalline Diamond (PCD)", 4.00, True,
        "Exceptional wear resistance on non-ferrous, non-reactive materials (aluminum, copper, "
        "composites). CHEMICALLY INCOMPATIBLE with ferrous, nickel-based, AND titanium "
        "materials -- carbon diffuses into iron/nickel, and titanium forms titanium carbide, "
        "at cutting temperatures (severe graphitization), causing rapid catastrophic edge "
        "failure. The system blocks this pairing rather than silently mis-predicting it.",
    ),
    "Ceramic": CoatingProfile(
        "Ceramic (Alumina / Silicon Nitride)", 2.50, False,
        "Very high hot hardness, excellent at high speed on cast iron and hardened steel, but "
        "brittle -- poor choice for interrupted cuts or heavy shock loading.",
    ),
    "CBN": CoatingProfile(
        "Cubic Boron Nitride (CBN)", 3.00, False,
        "Second-hardest material after diamond and, unlike diamond, thermally and chemically "
        "stable against iron and nickel -- the standard choice for hardened steel and "
        "superalloys at high cutting speed.",
    ),
}

# ---------------------------------------------------------------------------
# 4. TAYLOR MODEL VALIDITY ENVELOPE
# ---------------------------------------------------------------------------
TAYLOR_LIFE_MIN_MIN: float = 8.0
TAYLOR_LIFE_MAX_MIN: float = 300.0

# ---------------------------------------------------------------------------
# 5. SYNTHETIC TELEMETRY / TIME-SERIES SIMULATION PARAMETERS
# ---------------------------------------------------------------------------
SAMPLE_INTERVAL_MIN: float = 2.0
ROLLING_WINDOW_SAMPLES: int = 5
RUNTIME_STRETCH_RANGE: Tuple[float, float] = (0.50, 1.08)
INSTABILITY_EMA_DECAY: float = 0.75

RPM_NOISE_STD_FRAC: float = 0.012
FEED_NOISE_STD_FRAC: float = 0.02
DEPTH_NOISE_STD_FRAC: float = 0.03

INSTABILITY_EVENT_PROB: float = 0.05
INSTABILITY_NOISE_MULTIPLIER: float = 5.0

TOOL_INSTANCE_WEAR_FACTOR_STD: float = 0.09

# ---------------------------------------------------------------------------
# 6. FLANK WEAR (VB, mm) THREE-STAGE PROGRESSION MODEL
# ---------------------------------------------------------------------------
WEAR_BREAKIN_MAX_MM: float = 0.045
WEAR_BREAKIN_TAU_MIN: float = 6.0
WEAR_STEADY_COEF_MM: float = 0.22
WEAR_CATASTROPHIC_COEF_MM: float = 3.2
WEAR_CATASTROPHIC_ONSET_RATIO: float = 0.78
WEAR_INSTABILITY_COEF_MM: float = 0.006
WEAR_NOISE_STD_MM: float = 0.006

# ---------------------------------------------------------------------------
# 7. SCRAP RISK EVENT MODEL -- ISO 3685
# ---------------------------------------------------------------------------
SCRAP_VB_THRESHOLD_MM: float = 0.30          # ISO 3685 average flank-wear tool-life criterion
VISUAL_INSPECTION_VB_ALERT_MM: float = 0.25  # earlier operator alert, ahead of the hard threshold
SCRAP_LOGISTIC_STEEPNESS: float = 22.0
SCRAP_INSTABILITY_COEF: float = 1.4

# ---------------------------------------------------------------------------
# 8. DASHBOARD / DECISION-SUPPORT THRESHOLDS (Safety Guardrails)
# ---------------------------------------------------------------------------
WEAR_BAND_EDGES_MM: Tuple[float, float, float] = (0.15, VISUAL_INSPECTION_VB_ALERT_MM, SCRAP_VB_THRESHOLD_MM)
UNCERTAINTY_ALERT_SPREAD_MM: float = 0.08
RISK_TIER_EDGES: Tuple[float, float, float] = (0.20, 0.50, 0.80)

HEALTH_STATUS_OPTIMAL = "OPTIMAL"
HEALTH_STATUS_WARNING = "WARNING"
HEALTH_STATUS_CRITICAL = "CRITICAL"

RA_BAND_EDGES_UM: Tuple[float, float, float] = (1.6, 3.2, 6.3)

SPINDLE_RATED_POWER_KW: float = 15.0
SPINDLE_MECHANICAL_EFFICIENCY: float = 0.85

FEATURE_COLUMNS = [
    "material_code",
    "coolant_code",
    "coating_code",
    "spindle_rpm",
    "feed_rate_mm_rev",
    "axial_depth_mm",
    "radial_width_mm",
    "workpiece_diameter_mm",
    "num_flutes",
    "num_spindles",
    "nose_radius_mm",
    "cutting_speed_m_min",
    "runtime_min",
    "taylor_predicted_life_min",
    "taylor_ratio",
    "mrr_cm3_min",
    "cumulative_energy_kj",
    "feed_per_tooth_mm",
    "surface_roughness_ra_um",
    "spindle_load_pct",
    "rolling_rpm_std",
    "rolling_feed_std",
    "rolling_depth_std",
]

REGRESSION_TARGET = "flank_wear_mm"
CLASSIFICATION_TARGET = "scrap_event"
QUANTILES = (0.10, 0.50, 0.90)

MATERIAL_TO_CODE = {name: idx for idx, name in enumerate(MATERIALS.keys())}
CODE_TO_MATERIAL = {idx: name for name, idx in MATERIAL_TO_CODE.items()}
COOLANT_TO_CODE = {name: idx for idx, name in enumerate(COOLANT_CONDITIONS.keys())}
CODE_TO_COOLANT = {idx: name for name, idx in COOLANT_TO_CODE.items()}
COATING_TO_CODE = {name: idx for idx, name in enumerate(TOOL_COATINGS.keys())}
CODE_TO_COATING = {idx: name for name, idx in COATING_TO_CODE.items()}

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 9. FILE / ARTIFACT PATHS
# ---------------------------------------------------------------------------
DATA_DIR = "data"
SYNTHETIC_DATA_PATH = f"{DATA_DIR}/synthetic_telemetry.csv"
FEATURED_DATA_PATH = f"{DATA_DIR}/featured_telemetry.csv"
MODEL_DIR = "artifacts"
EXPORT_DIR = "exports"

N_SYNTHETIC_RUNS_DEFAULT = 1400

# ---------------------------------------------------------------------------
# 10. MODEL TIERS -- instant-boot default vs. opt-in full-precision upgrade
# ---------------------------------------------------------------------------
# Profiling: full-precision fit (1400 runs / 350 estimators / cv=4) takes
# ~24s, dominated by model fitting (not data generation). The fast-boot tier
# trades band precision for startup latency: fewer synthetic runs and a much
# smaller model, fit in well under a second -- verified empirically at
# ~0.4-0.5s across seeds. Conformal calibration still applies at this scale;
# with less calibration data it correctly yields a WIDER guaranteed-coverage
# band rather than an invalid one. This is genuine training on a micro
# corpus, not embedded pre-serialized weights: shipping stale binary weights
# would silently drift out of sync every time the feature schema changes
# (as it just did in this revision), which is a worse failure mode than a
# consistently-regenerated half-second fit against the CURRENT schema.
FAST_BOOT_BACKEND = "xgboost"
FAST_BOOT_N_RUNS = 50
FAST_BOOT_N_ESTIMATORS = 35
FAST_BOOT_MAX_DEPTH = 3
FAST_BOOT_CV_FOLDS = 2

FULL_PRECISION_N_RUNS = N_SYNTHETIC_RUNS_DEFAULT
FULL_PRECISION_N_ESTIMATORS = 350
FULL_PRECISION_MAX_DEPTH = 5
FULL_PRECISION_CV_FOLDS = 4

# ---------------------------------------------------------------------------
# 11. ACADEMIC BENCHMARK LITERATURE (Tab 3 Knowledge Base)
# ---------------------------------------------------------------------------
# Every entry below was independently checked against publisher/aggregator
# records rather than trusted as-supplied. 9 of the 11 DOIs originally
# provided for this list did not resolve to the stated paper (wrong final
# digits, a transposed digit, or an embedded-year mismatch) -- the papers
# themselves are real and correctly topically categorized, but the DOIs
# were not reliable. `doi` is populated only where this session's search
# confirmed the exact string against a publisher or multi-source citation
# match; `verified` is False where a paper is real but the exact DOI could
# not be pinned with confidence here, in which case the UI should NOT
# render a clickable link -- an unverified guess is worse than no link.
CITATIONS = [
    dict(category="Tool Wear Drivers & Physics-Guided ML", authors="Wang, Li, Zhao & Gao", year=2020,
         title="Physics guided neural network for machining tool wear prediction",
         journal="Journal of Manufacturing Systems 57:298-310", doi="10.1016/j.jmsy.2020.09.005", verified=True),
    dict(category="Tool Wear Drivers & Physics-Guided ML", authors="Rizal, Ghani, Nuawi & Haron", year=2013,
         title="Online tool wear prediction system in the turning process using an adaptive neuro-fuzzy inference system",
         journal="Applied Soft Computing 13(4):1960-1968", doi="10.1016/j.asoc.2012.11.043", verified=True),
    dict(category="Tool Wear Drivers & Physics-Guided ML", authors="Li, Liu, Hua, Gao & Maropoulos", year=2019,
         title="A novel method for accurately monitoring and predicting tool wear under varying cutting conditions based on meta-learning",
         journal="CIRP Annals 68(1):487-490", doi="10.1016/j.cirp.2019.03.010", verified=True),
    dict(category="Replacement Timing & Remaining Useful Life (RUL)", authors="An, Tao, Xu, El Mansori & Chen", year=2020,
         title="A data-driven model for milling tool remaining useful life prediction with convolutional and stacked LSTM network",
         journal="Measurement 154:107461", doi="10.1016/j.measurement.2019.107461", verified=True),
    dict(category="Replacement Timing & Remaining Useful Life (RUL)", authors="Yang, Guo, Huang, Chen, Li, Jiang & He", year=2019,
         title="Research on the milling tool wear and life prediction by establishing an integrated predictive model",
         journal="Measurement 145:178-189", doi="10.1016/j.measurement.2019.05.009", verified=True),
    dict(category="Replacement Timing & Remaining Useful Life (RUL)", authors="Yu, Liang, Tang & Liu", year=2017,
         title="A weighted hidden Markov model approach for continuous-state tool wear monitoring and tool life prediction",
         journal="Int J Adv Manuf Technol 91:201-211", doi="10.1007/s00170-016-9711-0", verified=True),
    dict(category="Uncertainty Quantification & Confidence Intervals", authors="Kong, Chen & Li", year=2018,
         title="Gaussian process regression for tool wear prediction",
         journal="Mechanical Systems and Signal Processing 104:556-574", doi=None, verified=False),
    dict(category="Uncertainty Quantification & Confidence Intervals", authors="Kong, Chen, Li & Tan", year=2019,
         title="Relevance vector machine for tool wear prediction",
         journal="Mechanical Systems and Signal Processing", doi=None, verified=False),
    dict(category="Uncertainty Quantification & Confidence Intervals", authors="Tobon-Mejia, Medjaher & Zerhouni", year=2012,
         title="CNC machine tool's wear diagnostic and prognostic by using dynamic Bayesian networks",
         journal="Mechanical Systems and Signal Processing 28:167-182", doi="10.1016/j.ymssp.2011.10.018", verified=True),
    dict(category="Quality Outcomes & Scrap Risk (Ra Surface Finish)", authors="Wang, Liu, Gao & Guo", year=2019,
         title="Heterogeneous data-driven hybrid machine learning for tool condition prognosis",
         journal="CIRP Annals 68(1):455-458", doi="10.1016/j.cirp.2019.03.007", verified=True),
    dict(category="Decision Support, Guardrails & Unsafe Operation Prevention", authors="Karam, Centobelli, D'Addona & Teti", year=2016,
         title="Online prediction of cutting tool life in turning via cognitive decision making",
         journal="Procedia CIRP 41:927-932", doi="10.1016/J.PROCIR.2016.01.002", verified=True),
    dict(category="Decision Support, Guardrails & Unsafe Operation Prevention", authors="Denkena et al.", year=2020,
         title="Simulation-based feed rate adaptation considering wear condition",
         journal="Procedia Manufacturing", doi=None, verified=False),
    dict(category="Decision Support, Guardrails & Unsafe Operation Prevention", authors="Pimenov, Bustillo, Wojciechowski, Sharma, Gupta & Kuntoglu", year=2022,
         title="Artificial intelligence systems for tool condition monitoring in machining: analysis and critical review",
         journal="Journal of Intelligent Manufacturing 34:2079-2121", doi="10.1007/s10845-022-01923-2", verified=True),
]
