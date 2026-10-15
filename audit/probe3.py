import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0,"/home/user/cnc-tool-wear-system")
import config
from data_generator import generate_dataset
from feature_engineering import build_modeling_table, engineer_features_for_single_observation
from ml_models import train_full_pipeline
from physics_engine import *
from physics_engine import taylor_ratio as taylor_ratio_fn

print("### A. FAST-BOOT TIER (what the live app trains on first load) ###")
t=time.perf_counter()
raw = generate_dataset(n_runs=config.FAST_BOOT_N_RUNS, seed=config.RANDOM_SEED)
tb  = build_modeling_table(raw)
fast, fm = train_full_pipeline(tb, backend=config.FAST_BOOT_BACKEND, n_estimators=config.FAST_BOOT_N_ESTIMATORS,
                               max_depth=config.FAST_BOOT_MAX_DEPTH, cv_folds=config.FAST_BOOT_CV_FOLDS, random_state=42)
print("fast-boot fit secs = %.2f | runs=%d rows=%d" % (time.perf_counter()-t, config.FAST_BOOT_N_RUNS, len(tb)))
print("fast coverage post-conformal = %.1f%% | band width = %.4f mm | conformal corr = %.4f mm | AUC %.3f Brier %.3f"
      % (fm["quantile_wear_model_post_conformal"]["q10_q90_empirical_coverage"]*100,
         fm["quantile_wear_model_post_conformal"]["mean_band_width_mm"], fm["conformal_correction_mm"],
         fm["scrap_risk_classifier"]["roc_auc"], fm["scrap_risk_classifier"]["brier_score"]))

print("\n### B. EXACT TAB-1 DASHBOARD SCENARIO (Inconel 718 / Flood / AlTiN) ###")
mat = config.MATERIALS["Nickel Superalloy (Inconel 718)"]
rpm, feed, ap, ae, D, Z, Ns, re_eps, tmin = 220, 0.085, 0.5, 0.65, 40.0, 4, 1, 0.8, 25.0
vc = cutting_speed_m_min(rpm, D)
bt = taylor_tool_life(vc, feed, ap, mat)
lm = apply_life_modifiers(bt.predicted_life_min, mat, "Flood", "AlTiN", allow_incompatible=True)
eff = lm.modified_life_min; tr = taylor_ratio(tmin, eff)
vf = table_feed_rate_mm_min(feed, rpm); mrr = material_removal_rate_cm3_min(ap, ae, vf, num_spindles=Ns)
ce = cutting_energy_kj(mrr, tmin, mat.specific_energy_j_mm3); fz = feed_per_tooth_mm(vf, rpm, Z)
ra = surface_roughness_ra_um(feed, re_eps); pk = cutting_power_kw(mrr, mat.specific_energy_j_mm3); ld = spindle_load_pct(pk)
print("Vc=%.1f m/min  T_base=%.0f min  T_eff=%.0f min  TaylorRatio=%.3f" % (vc, bt.predicted_life_min, eff, tr))
print("MRR=%.4f cm3/min  E_cum=%.0f kJ  fz=%.4f mm  Ra=%.2f um  P=%.2f kW  load=%.1f%%" % (mrr, ce, fz, ra, pk, ld))
obs = dict(material_code=config.MATERIAL_TO_CODE["Nickel Superalloy (Inconel 718)"], coolant_code=2, coating_code=2,
           spindle_rpm=rpm, feed_rate_mm_rev=feed, axial_depth_mm=ap, radial_width_mm=ae, workpiece_diameter_mm=D,
           num_flutes=Z, num_spindles=Ns, nose_radius_mm=re_eps, cutting_speed_m_min=vc, runtime_min=tmin,
           taylor_predicted_life_min=eff, taylor_ratio=tr, mrr_cm3_min=mrr, cumulative_energy_kj=ce,
           feed_per_tooth_mm=fz, surface_roughness_ra_um=ra, spindle_load_pct=ld,
           rolling_rpm_std=0.0, rolling_feed_std=0.0, rolling_depth_std=0.0)
X = pd.DataFrame([engineer_features_for_single_observation(obs)])[config.FEATURE_COLUMNS]
wp = fast[0]["wear_model"].predict(X).iloc[0] if isinstance(fast, tuple) else fast["wear_model"].predict(X).iloc[0]
sp = (fast[0] if isinstance(fast,tuple) else fast)["scrap_model"].predict_proba(X)[0]
print("\n>>> TAB-1 TILES THE OPERATOR SEES:")
print("    Vb Wear (Q50)      = %.3f mm   [Q10 %.3f / Q90 %.3f, dQ %.3f mm]" % (wp["q50"], wp["q10"], wp["q90"], wp["q90"]-wp["q10"]))
print("    Scrap Risk (P)     = %.1f %%  -> tier %s" % (sp*100, config.__dict__ and __import__('ml_models').ScrapRiskClassifier.risk_tier(sp)))
print("    Surface Roughness  = %.2f um" % ra)
print("    MRR (total)        = %.3f cm3/min" % mrr)
print("    Spindle Load       = %.1f %%" % ld)
print("    ISO 3685 threshold = %.2f mm  | visual alert = %.2f mm" % (config.SCRAP_VB_THRESHOLD_MM, config.VISUAL_INSPECTION_VB_ALERT_MM))
print("    band edges (green/amber/red) =", config.WEAR_BAND_EDGES_MM)

print("\n### C. README'S CLAIMED FULL-PRECISION NUMBERS (1200 runs, split by run_id) ###")
raw2 = generate_dataset(n_runs=1200, seed=config.RANDOM_SEED); tb2 = build_modeling_table(raw2)
print("corpus: %d runs / %d samples (README says 1,200 runs / ~54k samples)" % (raw2.run_id.nunique(), len(tb2)))
t=time.perf_counter()
m2, met2 = train_full_pipeline(tb2, backend="xgboost", n_estimators=config.FULL_PRECISION_N_ESTIMATORS,
                               max_depth=config.FULL_PRECISION_MAX_DEPTH, cv_folds=config.FULL_PRECISION_CV_FOLDS)
print("full-precision fit secs = %.1f (README/config claim ~24s)" % (time.perf_counter()-t))
pc = met2["quantile_wear_model_post_conformal"]; pr = met2["quantile_wear_model_pre_conformal"]
print("post-conformal coverage = %.1f%% (README claims ~83%%) | pre = %.1f%% (README claims ~68-73%%)" % (pc["q10_q90_empirical_coverage"]*100, pr["q10_q90_empirical_coverage"]*100))
print("mean band width = %.4f mm | conformal corr = %.4f mm" % (pc["mean_band_width_mm"], met2["conformal_correction_mm"]))
print("scrap ROC-AUC = %.3f (README ~0.83) | Brier = %.4f (README ~0.074)" % (met2["scrap_risk_classifier"]["roc_auc"], met2["scrap_risk_classifier"]["brier_score"]))
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
ids = tb2.run_id.unique(); _, te = train_test_split(ids, test_size=0.2, random_state=42)
ted = tb2[tb2.run_id.isin(te)]; pp = m2["wear_model"].predict(ted[config.FEATURE_COLUMNS])
print("Q50 MAE = %.4f mm | Q50 RMSE = %.4f mm  <-- NOT reported anywhere in the app or README"
      % (mean_absolute_error(ted.flank_wear_mm, pp["q50"]), np.sqrt(mean_squared_error(ted.flank_wear_mm, pp["q50"]))))
print("wear-level 3-class accuracy vs Paper2 stages (0-0.05/0.05-0.15/0.15-0.2+):")
def stage(v): return 0 if v<0.05 else (1 if v<0.15 else 2)
acc = np.mean([stage(a)==stage(b) for a,b in zip(ted.flank_wear_mm, pp["q50"])])
print("   = %.1f %%  (Paper 2 A-LWDRL: 98.7%% | Paper 3 range 84-100%%)" % (acc*100))
