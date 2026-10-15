import sys, numpy as np, pandas as pd
sys.path.insert(0,"/home/user/cnc-tool-wear-system")
import config
from data_generator import generate_dataset
from feature_engineering import build_modeling_table
raw = generate_dataset(n_runs=300, seed=config.RANDOM_SEED); t = build_modeling_table(raw)
print("### TRAIN/SERVE SKEW ON THE 3 'INSTABILITY' FEATURES ###")
print("The live app derives these from a UI slider, NOT from measurement (app.py:649-651):")
print("   rolling_*_std = ((100 - process_steadiness)/100) * k * setpoint,  k = 0.08/0.10/0.15")
print("   default process_steadiness = 100%  ->  all three features = 0.0 exactly\n")
print("%-22s %10s %10s %10s %12s" % ("feature","train mean","train p50","train p95","% of train == 0"))
for c in ["rolling_rpm_std","rolling_feed_std","rolling_depth_std"]:
    print("%-22s %10.4f %10.4f %10.4f %11.1f%%" % (c, t[c].mean(), t[c].median(), t[c].quantile(.95), (t[c]==0).mean()*100))
print("\nSERVED value at the app default (steadiness=100, Inconel 718 @220rpm/f0.085/ap0.5):")
print("   rolling_rpm_std=0.0000  rolling_feed_std=0.0000  rolling_depth_std=0.0000")
print("   -> percentile of served value within training distribution: ~%.1fth (below %.1f%% of training rows)"
      % ((t.rolling_rpm_std<=0).mean()*100, (t.rolling_rpm_std<=0).mean()*100))
print("\nSERVED value at steadiness=0 (worst case, Al 6061 @3200rpm/f0.27/ap1.7):")
for c,k,s in [("rolling_rpm_std",0.08,3200),("rolling_feed_std",0.10,0.27),("rolling_depth_std",0.15,1.7)]:
    v=k*s; print("   %-20s = %.4f  -> training percentile %.1f%% (train max %.4f)" % (c, v, (t[c]<=v).mean()*100, t[c].max()))
print("\nCONCLUSION: at the shipped default (steadiness=100%) the model is queried in a region")
print("that only ~1.4% of training rows occupy -> the served prediction is an extrapolation,")
print("and the conformal coverage guarantee (marginal, exchangeability) does not hold there.")
