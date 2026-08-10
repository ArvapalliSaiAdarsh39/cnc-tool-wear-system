# CNC Tool Wear Prediction & Scrap Risk Signal System

A production-grade MVP for Industry 4.0 predictive maintenance: a hybrid
**first-principles physics + machine learning** system that estimates CNC
cutting-tool flank wear (with a statistically-calibrated uncertainty band)
and calibrated scrap-defect risk, explains every prediction with SHAP, and
narrates it into a plain-language operator bulletin via Azure OpenAI --
all wrapped in a shop-floor-styled Streamlit console.

**Decision support only.** This system does not control, stop, or override
any machine tool. See [`docs/TECHNICAL_REFERENCE.md`](docs/TECHNICAL_REFERENCE.md#part-2----extraordinary-capabilities) for the full guardrail specification.

---

## What's inside

| Layer | What it does |
|---|---|
| **Physics Engine** | Extended Taylor tool-life equation, Material Removal Rate, cumulative cutting energy -- see [`physics_engine.py`](physics_engine.py) |
| **Synthetic Data Generator** | Physics-informed, non-linear (3-stage) tool-wear simulation across Aluminum / Steel / Titanium / Inconel -- see [`data_generator.py`](data_generator.py) |
| **Feature Engineering** | Rolling-window process-instability features -- see [`feature_engineering.py`](feature_engineering.py) |
| **Quantile ML Core** | Q10/Q50/Q90 wear regressors (XGBoost or LightGBM), **conformally calibrated** for a real coverage guarantee, plus an isotonic-calibrated scrap-risk classifier -- see [`ml_models.py`](ml_models.py) |
| **Explainable AI** | Local SHAP attribution for both models -- see [`shap_engine.py`](shap_engine.py) |
| **Generative Narrative Agent** | Azure OpenAI (GPT-4o) 3-sentence operator bulletins, JSON-payload driven, with a deterministic offline fallback -- see [`llm_narrative_agent.py`](llm_narrative_agent.py) |
| **PDF Export** | One-run operational report via reportlab -- see [`pdf_report.py`](pdf_report.py) |
| **Streamlit Console** | Live sliders, wear/uncertainty band, risk gauge, SHAP charts, bulletin, CSV/PDF export -- see [`app.py`](app.py) |
| **CLI Training Pipeline** | Batch-trains and persists model artifacts -- see [`train_pipeline.py`](train_pipeline.py) |

Full architecture rationale: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Every equation, metric, and guardrail defined precisely: [`docs/TECHNICAL_REFERENCE.md`](docs/TECHNICAL_REFERENCE.md).

---

## Quick start

```bash
# 1. Create a virtual environment (Python 3.12+)
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) configure Azure OpenAI for live LLM bulletins
cp .env.template .env
# edit .env with your Azure OpenAI endpoint / API key / deployment name
# -- the app runs perfectly well without this; bulletins use an offline
#    fallback generator when Azure OpenAI isn't configured.

# 4. (Optional) run the offline batch training CLI to persist model artifacts
python train_pipeline.py --n-runs 1500 --backend xgboost

# 5. Launch the dashboard
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`). The
dashboard trains its own models on first load (cached for the session) even
if you skip step 4.

### Docker

```bash
docker build -t cnc-tool-wear-system .
docker run --rm -p 8501:8501 --env-file .env cnc-tool-wear-system
```

---

## Project structure

```
cnc_tool_wear_system/
├── app.py                      # Streamlit dashboard (entrypoint)
├── train_pipeline.py           # CLI batch-training entrypoint
├── config.py                   # Single source of truth: materials, thresholds, paths
├── physics_engine.py            # Taylor tool-life, MRR, cutting energy, Taylor Ratio
├── data_generator.py            # Physics-informed synthetic telemetry generator
├── feature_engineering.py       # Rolling-window instability features
├── ml_models.py                  # Quantile wear model (+conformal) & calibrated scrap classifier
├── shap_engine.py                # SHAP explainability for both models
├── llm_narrative_agent.py        # Azure OpenAI narrative agent + offline fallback
├── pdf_report.py                  # PDF operator-report export
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .env.template                  # Mock credentials/config -- copy to .env and fill in
├── .gitignore
├── .streamlit/config.toml         # Dashboard theme
└── docs/
    ├── ARCHITECTURE.md            # Full architecture blueprint & shop-floor/cloud positioning
    └── TECHNICAL_REFERENCE.md     # Every equation, metric, parameter, and guardrail defined
```

---

## Model quality (synthetic MVP corpus, held out by tool run)

Numbers below are from this repository's own `train_pipeline.py` run against
1,200 synthetic tool-life runs (~54k telemetry samples), split by `run_id`
so no tool's samples leak across train/calibration/test:

- **Wear quantile band, post-conformal calibration:** ~83% empirical
  Q10-Q90 coverage against an 80% target (raw, pre-calibration coverage was
  ~68-73% -- see `docs/TECHNICAL_REFERENCE.md` Section 1.7 for why the
  conformal step matters).
- **Scrap-risk classifier:** ROC-AUC ~0.83, Brier score ~0.074.

Re-run `python train_pipeline.py` any time to regenerate these against a
fresh synthetic corpus or, after the recalibration steps in
`docs/ARCHITECTURE.md` Section 5, against real shop telemetry.

---

## Disclaimer

The material constants (Taylor coefficients, specific cutting energy,
safe parameter envelopes) shipped in `config.py` are illustrative,
literature-directional defaults for a synthetic MVP -- not certified
metallurgical data for any specific insert grade, coating, or coolant
strategy. See `docs/ARCHITECTURE.md` Section 5 before using this system's
numeric output to inform real production decisions.
