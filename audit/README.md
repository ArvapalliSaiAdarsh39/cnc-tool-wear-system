# Audit reproduction harness

Supporting artefacts for [`docs/VALIDATION_REPORT.md`](../docs/VALIDATION_REPORT.md) —
the independent validation audit of this system against three reference papers:

| | Paper | DOI |
|---|---|---|
| **[P1]** | Dong & Li, *Online Detection of Turning Tool Wear Based on Machine Vision*, ASME JCISE 22(5) 050903, 2022 | `10.1115/1.4053919` |
| **[P2]** | Fan, *A CNC Machine Tool Wear Prediction Model Integrating Deep Reinforcement Learning and Edge Data Processing*, DAPIC 2026 | `10.23919/DAPIC.2026.00023` |
| **[P3]** | Munaro, Attanasio & Del Prete, *Tool Wear Monitoring with Artificial Intelligence Methods: A Review*, JMMP 7(4) 129, 2023 | `10.3390/jmmp7040129` |

Every quantitative claim in the validation report was produced by running this
repository's **own** code. Nothing was read off the README or estimated.

## What's here

| File | Produces |
|---|---|
| `probe.py` | Full-pipeline metrics (400 runs); **Q50 MAE/RMSE** (not computed anywhere else in this project); inference latency for the wear band, the scrap classifier and SHAP; joblib artifact sizes; simulated-VB distribution scored against [P2]'s stage boundaries *and* the system's own band edges; scrap-event rates |
| `probe2.py` | Label-recoverability / circularity analysis (Pearson & Spearman r of `taylor_ratio` vs true VB; the generator's irreducible noise floor; ROC-AUC of raw VB as a scrap classifier); confirmation that `instability_recent_ema` drives labels but is absent from `FEATURE_COLUMNS`; sampling-rate comparison vs [P2]; **feature-domain audit** showing RMS / mean / kurtosis / skewness / peak / variance / FFT / wavelet / EMD / STFT / PCA / spectrum are all absent from `FEATURE_COLUMNS` |
| `probe3.py` | **The shipped fast-boot tier exactly as the live app trains it** (50 runs / 35 estimators / depth 3 / cv 2), an exact Tab-1 dashboard scenario for Inconel 718 reproducing the five metric tiles an operator sees, and re-measurement of every numeric claim in the root `README.md` "Model quality" block at its stated 1,200-run configuration |
| `probe4.py` | Per-material `spindle_load_pct` / MRR / Vc table establishing that feature's degeneracy; feature-variance audit; a worked wear-progression curve by life fraction against the `config.py` §6 stage parameters |
| `probe5.py` | **Train/serve skew quantification** — the training distribution of the three `rolling_*_std` features versus the values `app.py:649-651` actually serves at `process_steadiness` = 100 % and 0 %, expressed as training percentiles |
| `pdf_extract.py` | The `pypdf` extraction script used on the three supplied PDFs |
| `txt/paper{1,2,3}_*.txt` | Page-marked extracted text of [P1] (5 pp.), [P2] (5 pp.), [P3] (48 pp.) — retained so every citation in the report can be checked against the source |

## How to re-run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pypdf      # pypdf only needed for pdf_extract.py

# from the repository root — the probes insert the repo root on sys.path themselves
python audit/probe.py      # ~15 s
python audit/probe2.py     # ~5 s
python audit/probe3.py     # ~25 s  (includes the full-precision 1,200-run fit)
python audit/probe4.py     # ~5 s
python audit/probe5.py     # ~5 s
```

## Provenance of the measured numbers

| | |
|---|---|
| Repo state audited | `main` @ `0ee8a884`, pushed 2026-08-12T05:52:07Z. Verified via GitHub API `git/trees/main?recursive=1` to contain exactly the 17 tracked files, and **not** to contain `docs/` or `.streamlit/` |
| Audit environment | Python 3.11.2; `numpy pandas scikit-learn xgboost lightgbm joblib shap streamlit plotly reportlab python-dotenv openai` |
| Application executed | `streamlit run app.py --server.address 0.0.0.0 --server.port 8501` → HTTP 200, no errors |
| Seed | `config.RANDOM_SEED = 42`, XGBoost backend throughout |
| Date | 15 October 2026 |

⚠️ **Your numbers may differ slightly from the report's.** The audit installed
the dependency stack unpinned-to-latest-compatible rather than the exact pins in
`requirements.txt` (the sandbox had Python 3.11, while `requirements.txt`
specifies 3.12+). XGBoost/LightGBM/sklearn versions affect tree construction, so
re-running under the exact pinned stack may shift metrics in the third decimal
place. The *findings* do not depend on that precision: the conformal correction
being exactly `0.0000` on the fast-boot tier, the served `rolling_*_std` values
being exactly `0.0` at the default slider position, MAE/RMSE being absent from
`QuantileWearModel.evaluate()`, and the zero repository-wide matches for any
economic term are all **structural** facts, not measured ones.

## Not reachable from the audit sandbox

`https://cnc-tool-wear-system.streamlit.app/` — egress was restricted to
github.com, api.github.com, codeload.github.com, registry.npmjs.org, pypi.org and
files.pythonhosted.org. This was mitigated rather than papered over: the deployed
commit was confirmed identical to the audited checkout and then **run locally**, so
the measurements are of the deployed application's own source. If Community Cloud
is serving a cached older process its in-memory metrics could differ; the commit it
deploys from is the source of truth audited here.
