"""
Phase 3 — Async quality verification loop.

After a response is returned to the user, we queue a background job that:
  1. Sends the same prompt to the verifier model (the highest-tier model)
  2. Scores agreement between the cheap model's output and the verifier's output
  3. If they diverge past a threshold, logs a routing failure and (optionally)
     re-runs with the higher-tier model, logging the escalation event
  4. Appends the failure to a feedback file that the classifier retrainer reads
"""
import json
import os
import time
from difflib import SequenceMatcher

from app.config import settings
from app.models_registry import get_model
from app.providers import send_request
from app.routing import verifier_model
from app import db

FEEDBACK_PATH = "./data/routing_failures.jsonl"


def _similarity(a: str, b: str) -> float:
    """Simple, dependency-free agreement score. A TF-IDF cosine or embedding
    distance would be more accurate in production; this keeps the portfolio
    project dependency-light while still catching gross divergence."""
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def _record_failure(prompt: str, cheap_model: str, tier: int, similarity: float):
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with open(FEEDBACK_PATH, "a") as f:
        f.write(
            json.dumps(
                {
                    "prompt": prompt,
                    "routed_model": cheap_model,
                    "original_tier": tier,
                    "similarity": similarity,
                    "timestamp": time.time(),
                    # the classifier under-scored this prompt; feed it back as tier 3
                    "corrected_tier": 3,
                }
            )
            + "\n"
        )


def verify_and_maybe_escalate(
    request_id: str,
    prompt: str,
    cheap_output: str,
    routed_model_name: str,
    tier: int,
):
    """Designed to be run as a FastAPI BackgroundTask — runs after the response
    has already been sent to the user, so it never adds latency to the request."""
    if not settings.VERIFIER_ENABLED:
        return

    v_model = verifier_model()
    if v_model.name == routed_model_name:
        # already routed to the top-tier model, nothing to verify against
        db.log_verification(request_id, quality_score=1.0, escalated=False)
        return

    reference = send_request(prompt, v_model)
    score = _similarity(cheap_output, reference.text)

    escalated = False
    if score < settings.ESCALATION_SIMILARITY_THRESHOLD:
        _record_failure(prompt, routed_model_name, tier, score)
        if settings.AUTO_ESCALATE:
            escalated = True
            db.log_verification(
                request_id,
                quality_score=score,
                escalated=True,
                escalated_model=v_model.name,
                escalated_cost_usd=reference.cost_usd,
            )
            return

    db.log_verification(request_id, quality_score=score, escalated=escalated)
