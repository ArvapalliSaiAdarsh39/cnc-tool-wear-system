import sys, numpy as np, pandas as pd
sys.path.insert(0,"/home/user/cnc-tool-wear-system")
import config
from data_generator import generate_dataset
from feature_engineering import build_modeling_table
raw = generate_dataset(n_runs=200, seed=config.RANDOM_SEED); t = build_modeling_table(raw)
print("=== LABEL-RECOVERABILITY / CIRCULARITY CHECK ===")
print("samples per tool run: mean %.1f (min %d max %d) | sample interval %.1f min"
      % (t.groupby('run_id').size().mean(), t.groupby('run_id').size().min(), t.groupby('run_id').size().max(), config.SAMPLE_INTERVAL_MIN))
# how deterministic is the generator's VB given taylor_ratio alone?
sub = t[["taylor_ratio","flank_wear_mm"]].dropna()
print("Pearson r(taylor_ratio, VB) = %.4f" % sub.corr().iloc[0,1])
print("Spearman rho                = %.4f" % sub.corr(method='spearman').iloc[0,1])
# residual spread of VB around a smooth function of taylor_ratio (i.e. irreducible aleatoric noise)
x=sub.taylor_ratio.values; y=sub.flank_wear_mm.values
c=np.polyfit(x,y,6); resid=y-np.polyval(c,x)
print("irreducible VB noise around poly6(taylor_ratio): std %.4f mm, MAE %.4f mm" % (resid.std(), np.abs(resid).mean()))
print("  -> the generator's own noise floor (WEAR_NOISE_STD_MM=%.3f + break-in + instability) sets the MAE limit" % config.WEAR_NOISE_STD_MM)
# scrap label recoverability
print("\nscrap_event vs VB: point-biserial r = %.4f" % t[["flank_wear_mm","scrap_event"]].corr().iloc[0,1])
print("AUC of RAW VB as a scrap classifier (no ML):", end=" ")
from sklearn.metrics import roc_auc_score
print("%.4f" % roc_auc_score(t.scrap_event, t.flank_wear_mm))
print("  (the trained ML classifier's ROC-AUC was ~0.856 -> ML adds little over the VB threshold itself)")
print("\ninstability_recent_ema (a true label driver) present in FEATURE_COLUMNS?", "instability_recent_ema" in config.FEATURE_COLUMNS)
print("instability_event present?", "instability_event" in config.FEATURE_COLUMNS)
print("=> scrap labels depend on instability_recent_ema, which is NOT a model feature.")
# sampling frequency comparison
print("\n=== SAMPLING RATE ===")
print("system telemetry: 1 sample / %.0f min = %.4f Hz" % (config.SAMPLE_INTERVAL_MIN, 1/(config.SAMPLE_INTERVAL_MIN*60)))
print("Paper 2 PHM2010: 5000 Hz | Paper 2 real 45# steel: 1000 Hz")
print("ratio: system samples %.0fx slower than Paper2's real dataset" % (1000/(1/(config.SAMPLE_INTERVAL_MIN*60))))
# feature domain audit
print("\n=== FEATURE DOMAIN AUDIT (Paper 3 Sec 4.5.2) ===")
td = [c for c in config.FEATURE_COLUMNS if c.startswith("rolling_")]
print("time-domain statistical features present:", td)
for f in ["rms","mean","kurtosis","skewness","peak","variance","fft","wavelet","emd","stft","pca","spectrum"]:
    hits=[c for c in config.FEATURE_COLUMNS if f in c.lower()]
    print("  %-10s -> %s" % (f, hits if hits else "ABSENT"))
