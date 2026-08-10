"""
llm_narrative_agent.py
========================
Generative AI Narrative Agent.

Translates the numeric outputs of the physics engine, quantile wear model,
scrap-risk classifier, and SHAP explainer into a concise, natural-language
operator bulletin via Azure OpenAI (GPT-4o).

Contract with the model: EXACTLY three sentences --
  1. Current-state assessment (wear level + scrap risk tier, in plain language)
  2. The key driver(s), grounded in the SHAP attributions (never invented)
  3. A recommended next action, bounded by the system's safety guardrails

Safety guardrails enforced in the system prompt itself (not just hoped for):
  * The bulletin must never instruct or imply an automatic machine
    shutdown or parameter override -- this system is decision support only.
  * If the uncertainty band is wide, the bulletin must say so and recommend
    a manual visual inspection rather than asserting false precision.
  * The bulletin must not invent numbers, causes, or SHAP drivers beyond
    what is in the supplied JSON payload.

Resilience: bulletin generation must never be a single point of failure for
a safety-adjacent dashboard. If Azure OpenAI credentials are absent, the
endpoint is unreachable, the call errors, or the call times out, this module
transparently falls back to a deterministic, rule-based bulletin generator
that follows the exact same 3-sentence contract using only arithmetic on the
same payload -- no network dependency, no LLM, but never a blank panel.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

import config
from ml_models import ScrapRiskClassifier
from shap_engine import SHAPExplanation

logger = logging.getLogger(__name__)

load_dotenv()  # populate os.environ from a local .env file if present, no-op otherwise

SYSTEM_PROMPT = """You are an operational bulletin writer embedded in a CNC tool-wear \
predictive-maintenance system. You receive a single JSON payload describing one \
machining tool's current telemetry, physics-derived tool-life estimate, ML wear \
prediction with uncertainty band, calibrated scrap-risk probability, and the top \
SHAP feature drivers behind those predictions.

Write EXACTLY three sentences, in this order, and nothing else (no preamble, no \
headers, no bullet points, no markdown):
1. State the current flank-wear estimate (with its uncertainty band if the band is \
   wide) and the scrap-risk tier, in plain shop-floor language.
2. Name the one or two SHAP drivers most responsible, using the feature display \
   names given in the payload -- never invent a driver that is not listed.
3. Recommend one concrete next action for the operator or process engineer.

Hard constraints:
- This system provides DECISION SUPPORT ONLY. Never instruct, imply, or suggest an \
  automatic machine shutdown, automatic parameter change, or that the system itself \
  will take any action. Any action is something a human does.
- If `uncertainty_flag` is true in the payload, sentence 1 or 3 must recommend a \
  manual visual inspection of the cutting edge rather than asserting a confident \
  single number.
- If `coating_incompatibility_warning` is non-null, sentence 1 MUST lead with that \
  incompatibility rather than the normal wear narrative -- an incompatible \
  coating/material pairing means imminent chemical edge failure, not gradual wear, \
  and this takes priority over every other observation.
- Never state a number that is not present in or directly computable from the payload.
- Do not use exclamation points or alarmist language. Write like an experienced \
  reliability engineer, not a warning siren.
"""


@dataclass
class BulletinResult:
    bulletin_text: str
    source: str  # "azure_openai" or "fallback_template"
    model_name: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    payload: dict = field(default_factory=dict)


def build_payload(
    material: str,
    coolant_label: str,
    coating_label: str,
    coating_compatible: bool,
    coating_warning: Optional[str],
    runtime_min: float,
    taylor_predicted_life_min: float,
    taylor_ratio_value: float,
    taylor_extrapolated: bool,
    wear_quantiles: dict,          # {"q10":..., "q50":..., "q90":...}
    scrap_probability: float,
    scrap_risk_tier: str,
    wear_shap: SHAPExplanation,
    scrap_shap: SHAPExplanation,
) -> dict:
    """
    Assembles the single structured JSON payload sent to the LLM (and reused
    verbatim by the offline fallback generator, so both paths are always
    looking at identical numbers).
    """
    band_width = wear_quantiles["q90"] - wear_quantiles["q10"]
    uncertainty_flag = band_width > config.UNCERTAINTY_ALERT_SPREAD_MM

    return {
        "material": material,
        "coolant_condition": coolant_label,
        "tool_coating": coating_label,
        "coating_compatible": bool(coating_compatible),
        "coating_incompatibility_warning": coating_warning,
        "runtime_min": round(runtime_min, 1),
        "taylor_predicted_life_min": round(taylor_predicted_life_min, 1),
        "taylor_ratio": round(taylor_ratio_value, 3),
        "taylor_model_extrapolated": bool(taylor_extrapolated),
        "wear_estimate_mm": {
            "q10_optimistic": round(wear_quantiles["q10"], 4),
            "q50_expected": round(wear_quantiles["q50"], 4),
            "q90_pessimistic": round(wear_quantiles["q90"], 4),
            "band_width_mm": round(band_width, 4),
        },
        "uncertainty_flag": bool(uncertainty_flag),
        "scrap_risk": {
            "probability": round(scrap_probability, 3),
            "tier": scrap_risk_tier,
        },
        "wear_shap_top_drivers": wear_shap.as_records(n=3),
        "scrap_shap_top_drivers": scrap_shap.as_records(n=3),
        "scrap_vb_threshold_mm": config.SCRAP_VB_THRESHOLD_MM,
    }


def _fallback_bulletin(payload: dict) -> str:
    """
    Deterministic, template-based 3-sentence bulletin used whenever Azure
    OpenAI is unavailable. Follows the identical contract as the LLM path:
    sentence 1 = state, sentence 2 = drivers, sentence 3 = action. Every
    number and driver name is read directly out of `payload` -- this
    function performs no inference of its own beyond formatting.
    """
    wear = payload["wear_estimate_mm"]
    tier = payload["scrap_risk"]["tier"]
    prob_pct = payload["scrap_risk"]["probability"] * 100

    if payload.get("coating_incompatibility_warning"):
        s1 = (
            f"This run pairs {payload['tool_coating']} tooling with {payload['material']}, a "
            f"combination that fails by rapid edge chemistry breakdown rather than the gradual "
            f"wear curve modeled below."
        )
        s2 = payload["coating_incompatibility_warning"]
        s3 = "Recommend switching to CBN or Ceramic tooling for this material before continuing production."
        return f"{s1} {s2} {s3}"

    if payload["uncertainty_flag"]:
        s1 = (
            f"Expected flank wear on this {payload['material']} run is {wear['q50_expected']:.3f} mm, "
            f"but the model's uncertainty band ({wear['q10_optimistic']:.3f}-{wear['q90_pessimistic']:.3f} mm) "
            f"is wide and scrap risk is currently rated {tier} ({prob_pct:.0f}%)."
        )
    else:
        s1 = (
            f"Expected flank wear on this {payload['material']} run is {wear['q50_expected']:.3f} mm "
            f"with a tight uncertainty band, and scrap risk is currently rated {tier} ({prob_pct:.0f}%)."
        )

    top_wear_drivers = payload["wear_shap_top_drivers"][:2]
    driver_phrases = [d["display_name"] for d in top_wear_drivers if d["shap_value"] > 0] or \
                      [d["display_name"] for d in top_wear_drivers]
    if driver_phrases:
        s2 = f"The main contributor{'s are' if len(driver_phrases) > 1 else ' is'} " + \
             " and ".join(driver_phrases) + "."
    else:
        s2 = "No single feature dominates the current estimate; drivers are evenly distributed."

    if payload["uncertainty_flag"]:
        s3 = "Recommend a manual visual inspection of the cutting edge before continuing this run."
    elif tier in ("High", "Critical"):
        s3 = "Recommend scheduling a tool change or inspection at the next natural break in the cycle."
    elif tier == "Medium":
        s3 = "Recommend continued monitoring and a routine visual check at the next tool-change interval."
    else:
        s3 = "No action required beyond standard periodic monitoring."

    return f"{s1} {s2} {s3}"


class NarrativeAgent:
    """
    Azure OpenAI (GPT-4o) wrapper. Reads connection details from environment
    variables (see .env.template):

        AZURE_OPENAI_ENDPOINT
        AZURE_OPENAI_API_KEY
        AZURE_OPENAI_API_VERSION
        AZURE_OPENAI_DEPLOYMENT_NAME

    If any required variable is missing, or the API call fails/times out for
    any reason, `generate_bulletin` transparently returns the deterministic
    fallback bulletin instead of raising -- see module docstring.
    """

    def __init__(self, timeout_seconds: float = 12.0):
        self.timeout_seconds = timeout_seconds
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self._client = None

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key and self.deployment_name)

    def _get_client(self):
        if self._client is None:
            from openai import AzureOpenAI  # imported lazily so the module loads with no `openai` extras issues

            self._client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                timeout=self.timeout_seconds,
            )
        return self._client

    def generate_bulletin(self, payload: dict) -> BulletinResult:
        if not self.is_configured:
            logger.info("Azure OpenAI not configured; using offline fallback bulletin.")
            return BulletinResult(
                bulletin_text=_fallback_bulletin(payload),
                source="fallback_template",
                error="Azure OpenAI credentials not configured (see .env.template).",
                payload=payload,
            )

        start = time.perf_counter()
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
                max_tokens=220,
                temperature=0.2,
            )
            bulletin_text = response.choices[0].message.content.strip()
            latency_ms = (time.perf_counter() - start) * 1000

            return BulletinResult(
                bulletin_text=bulletin_text,
                source="azure_openai",
                model_name=self.deployment_name,
                latency_ms=latency_ms,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure must fall back, never crash the dashboard
            logger.warning("Azure OpenAI call failed (%s); using offline fallback bulletin.", exc)
            return BulletinResult(
                bulletin_text=_fallback_bulletin(payload),
                source="fallback_template",
                error=str(exc),
                payload=payload,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo_payload = {
        "material": "Titanium Alloy (Ti-6Al-4V)",
        "coolant_condition": "Flood Coolant",
        "tool_coating": "CBN (Cubic Boron Nitride)",
        "coating_compatible": True,
        "coating_incompatibility_warning": None,
        "runtime_min": 142.0,
        "taylor_predicted_life_min": 168.4,
        "taylor_ratio": 0.843,
        "taylor_model_extrapolated": False,
        "wear_estimate_mm": {"q10_optimistic": 0.221, "q50_expected": 0.268, "q90_pessimistic": 0.331, "band_width_mm": 0.110},
        "uncertainty_flag": True,
        "scrap_risk": {"probability": 0.62, "tier": "High"},
        "wear_shap_top_drivers": [
            {"feature": "taylor_ratio", "display_name": "Taylor Ratio (runtime / predicted life)", "shap_value": 0.041, "feature_value": 0.843, "direction": "increases"},
            {"feature": "rolling_feed_std", "display_name": "Feed Rate Instability (rolling std)", "shap_value": 0.018, "feature_value": 0.006, "direction": "increases"},
        ],
        "scrap_shap_top_drivers": [
            {"feature": "taylor_ratio", "display_name": "Taylor Ratio (runtime / predicted life)", "shap_value": 0.88, "feature_value": 0.843, "direction": "increases"},
        ],
        "scrap_vb_threshold_mm": 0.30,
    }

    agent = NarrativeAgent()
    print("Azure OpenAI configured:", agent.is_configured, "(expected False in this sandbox)")
    result = agent.generate_bulletin(demo_payload)
    print("\nsource:", result.source)
    print("error:", result.error)
    print("\nBULLETIN:\n", result.bulletin_text)

    # Also exercise the "no uncertainty flag / low risk" branch of the fallback
    calm_payload = dict(demo_payload)
    calm_payload["uncertainty_flag"] = False
    calm_payload["scrap_risk"] = {"probability": 0.05, "tier": "Low"}
    calm_payload["wear_estimate_mm"] = {"q10_optimistic": 0.09, "q50_expected": 0.10, "q90_pessimistic": 0.11, "band_width_mm": 0.02}
    print("\n--- calm scenario ---")
    print(_fallback_bulletin(calm_payload))

    # Exercise the incompatible-coating short-circuit
    bad_payload = dict(demo_payload)
    bad_payload["tool_coating"] = "Polycrystalline Diamond (PCD)"
    bad_payload["coating_compatible"] = False
    bad_payload["coating_incompatibility_warning"] = (
        "Polycrystalline Diamond (PCD) is chemically incompatible with Titanium Alloy (Ti-6Al-4V) (ferrous/nickel/titanium). "
        "Carbon diffusion or carbide formation at the cutting interface causes rapid edge "
        "failure, not gradual wear -- select CBN or Ceramic for this material instead."
    )
    print("\n--- incompatible coating scenario ---")
    print(_fallback_bulletin(bad_payload))
