import sys, numpy as np, pandas as pd
sys.path.insert(0,"/home/user/cnc-tool-wear-system")
import config
from data_generator import generate_dataset
from feature_engineering import build_modeling_table
raw = generate_dataset(n_runs=300, seed=config.RANDOM_SEED); t = build_modeling_table(raw)
print("### SPINDLE LOAD / POWER FEATURE DEGENERACY (23-feature audit) ###")
sl = t.spindle_load_pct
print("spindle_load_pct: mean %.3f%% | median %.3f%% | p95 %.3f%% | max %.2f%% | frac < 1%% = %.1f%%"
      % (sl.mean(), sl.median(), sl.quantile(.95), sl.max(), (sl<1).mean()*100))
print("  -> SPINDLE_RATED_POWER_KW = %.1f kW, eff = %.2f (config)" % (config.SPINDLE_RATED_POWER_KW, config.SPINDLE_MECHANICAL_EFFICIENCY))
print("\nper-material mean spindle load / MRR:")
g = t.groupby("material").agg(load=("spindle_load_pct","mean"), mrr=("mrr_cm3_min","mean"), vc=("cutting_speed_m_min","mean")).round(3)
print(g.sort_values("load").to_string())
print("\n### FEATURE VARIANCE / USEFULNESS ###")
X = t[config.FEATURE_COLUMNS]
nun = X.nunique()
print("features with <=1 unique value in corpus:", [c for c in nun.index if nun[c]<=1])
print("near-constant (unique<=3):", [(c,int(nun[c])) for c in nun.index if nun[c]<=3])
print("\nrolling_* std features: mean / max")
for c in ["rolling_rpm_std","rolling_feed_std","rolling_depth_std"]:
    print("  %-20s mean %.5f  max %.5f  (frac==0: %.1f%%)" % (c, t[c].mean(), t[c].max(), (t[c]==0).mean()*100))
print("  NOTE: in the live app these are set by a UI slider (process_steadiness), not measured:")
print("        rolling_rpm_std = ((100-steadiness)/100) * 0.08 * rpm  [app.py:649-651]")
print("        at the default steadiness=100% all three are exactly 0.0")
print("\n### WEAR PROGRESSION SHAPE (compare Paper 2 Fig.2) ###")
r = t[t.run_id==t.run_id.iloc[0]].sort_values("sample_index")
big = t.groupby("run_id").filter(lambda d: len(d)>40)
one = big[big.run_id==big.run_id.unique()[3]].sort_values("sample_index")
print("example run: runtime %.0f->%.0f min, VB %.3f->%.3f mm, taylor_life %.0f" % (one.runtime_min.min(), one.runtime_min.max(), one.flank_wear_mm.min(), one.flank_wear_mm.max(), one.taylor_predicted_life_min.iloc[0]))
q = one.assign(frac=one.runtime_min/one.taylor_predicted_life_min.iloc[0])
for lo in [0,0.2,0.4,0.6,0.78,0.9,1.0]:
    s=q[(q.frac>=lo)&(q.frac<lo+0.1)]
    if len(s): print("   life-fraction %.2f-%.2f : mean VB %.3f mm" % (lo, lo+0.1, s.flank_wear_mm.mean()))
print("\nconfig wear-stage params: breakin_max %.3f mm (tau %.1f min) | steady coef %.2f | catastrophic coef %.1f onset ratio %.2f"
      % (config.WEAR_BREAKIN_MAX_MM, config.WEAR_BREAKIN_TAU_MIN, config.WEAR_STEADY_COEF_MM, config.WEAR_CATASTROPHIC_COEF_MM, config.WEAR_CATASTROPHIC_ONSET_RATIO))
print("  -> 3-stage shape (break-in / steady / catastrophic) QUALITATIVELY MATCHES Paper 2 & classic wear curves")
print("  -> but break-in ceiling 0.045 mm ~= Paper 2 'initial' stage top (0.05 mm)  [MATCH]")
print("  -> catastrophic onset at 0.78*T and coef 3.2 drives VB past Paper 2's 0.2 mm 'final' ceiling up to ISO 0.3 mm [DEVIATION]")
