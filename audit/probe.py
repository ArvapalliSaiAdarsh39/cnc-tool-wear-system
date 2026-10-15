import sys, time, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/cnc-tool-wear-system")
import config
from data_generator import generate_dataset
from feature_engineering import build_modeling_table
from ml_models import train_full_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

t0=time.perf_counter()
raw = generate_dataset(n_runs=400, seed=config.RANDOM_SEED)
table = build_modeling_table(raw)
print("gen+feat secs %.1f" % (time.perf_counter()-t0), "rows", len(table), "runs", raw.run_id.nunique())
t1=time.perf_counter()
models, metrics = train_full_pipeline(table, backend="xgboost", n_estimators=200, max_depth=5, cv_folds=3)
print("train secs %.1f" % (time.perf_counter()-t1))
print(json.dumps({k:v for k,v in metrics.items() if isinstance(v,dict)}, indent=1, default=float))

# --- MAE / RMSE of the Q50 point estimate (Paper 2 benchmark metric) ---
from sklearn.model_selection import train_test_split
run_ids = table["run_id"].unique()
tr_ids, te_ids = train_test_split(run_ids, test_size=0.2, random_state=42)
te = table[table.run_id.isin(te_ids)]
pred = models["wear_model"].predict(te[config.FEATURE_COLUMNS])
y = te[config.REGRESSION_TARGET].values
mae = mean_absolute_error(y, pred["q50"]); rmse = float(np.sqrt(mean_squared_error(y, pred["q50"])))
print("\n=== Q50 POINT-ESTIMATE ERROR ON HELD-OUT RUNS (synthetic ground truth) ===")
print("MAE  = %.4f mm   (Paper2 benchmark 0.023-0.028 mm)" % mae)
print("RMSE = %.4f mm   (Paper2 benchmark 0.031-0.036 mm)" % rmse)

# --- inference latency (Paper 2 benchmark 12.8 ms) ---
X1 = te[config.FEATURE_COLUMNS].head(1)
for _ in range(5): models["wear_model"].predict(X1)
t=time.perf_counter(); N=200
for _ in range(N): models["wear_model"].predict(X1)
lat=(time.perf_counter()-t)/N*1000
t=time.perf_counter()
for _ in range(N): models["scrap_model"].predict_proba(X1)
lat2=(time.perf_counter()-t)/N*1000
print("wear-band inference latency = %.2f ms/sample (3 quantile models + conformal)  [Paper2: 12.8 ms]" % lat)
print("scrap-proba inference latency = %.2f ms/sample (4-fold calibrated ens.)      " % lat2)

# --- inference latency incl. SHAP ---
from shap_engine import explain_wear_prediction, explain_scrap_risk
t=time.perf_counter()
for _ in range(20):
    explain_wear_prediction(models["wear_model"], X1); explain_scrap_risk(models["scrap_model"], X1)
print("SHAP (wear+scrap) latency = %.1f ms/sample" % ((time.perf_counter()-t)/20*1000))

# --- model size (Paper 2 benchmark 2.3 MB) ---
import joblib, os, tempfile
d = tempfile.mkdtemp()
joblib.dump(models["wear_model"], d+"/w.joblib"); joblib.dump(models["scrap_model"], d+"/s.joblib")
print("artifact sizes: wear %.2f MB, scrap %.2f MB  [Paper2: 2.3 MB total]" % (os.path.getsize(d+"/w.joblib")/1e6, os.path.getsize(d+"/s.joblib")/1e6))

# --- wear-stage distribution vs Paper 2 stages ---
vb = table["flank_wear_mm"].values
print("\n=== SIMULATED VB DISTRIBUTION (all samples) ===")
for lo,hi,name in [(0,0.05,"P2 Initial 0-0.05"),(0.05,0.15,"P2 Intermediate .05-.15"),(0.15,0.2,"P2 Final .15-.2"),(0.2,0.3,"BEYOND P2 / below ISO .2-.3"),(0.3,99,">= ISO 3685 0.3")]:
    frac=((vb>=lo)&(vb<hi)).mean()*100; print("  %-32s %5.1f%%" % (name, frac))
print("  max VB %.3f mm | p50 %.3f | p90 %.3f | p99 %.3f" % (vb.max(), np.percentile(vb,50), np.percentile(vb,90), np.percentile(vb,99)))
# system band edges
print("\n=== SYSTEM'S OWN 4-BAND EDGES (config.WEAR_BAND_EDGES_MM) ===", config.WEAR_BAND_EDGES_MM, " scrap thr", config.SCRAP_VB_THRESHOLD_MM, " visual alert", config.VISUAL_INSPECTION_VB_ALERT_MM)
b = config.WEAR_BAND_EDGES_MM
for lo,hi,name in [(0,b[0],"Band1 0-0.15"),(b[0],b[1],"Band2 .15-.25"),(b[1],b[2],"Band3 .25-.30"),(b[2],99,"Band4 >=0.30")]:
    print("  %-18s %5.1f%%" % (name, ((vb>=lo)&(vb<hi)).mean()*100))

# --- scrap rate + label circularity check ---
print("\n=== SCRAP EVENT MODEL ===")
print("overall scrap_event rate = %.4f" % table["scrap_event"].mean())
print("scrap rate when VB<0.30  = %.4f" % table.loc[table.flank_wear_mm<0.30,"scrap_event"].mean())
print("scrap rate when VB>=0.30 = %.4f" % table.loc[table.flank_wear_mm>=0.30,"scrap_event"].mean())
# how much of scrap prob is explained by VB alone
from numpy.polynomial import polynomial as P
print("\n=== FEATURE COUNT ===", len(config.FEATURE_COLUMNS))
