"""
Phase 2 (routing map) + glue between the classifier and the model registry.
"""
import os
import yaml
import joblib
import numpy as np

from app.config import settings
from app.models_registry import get_model, ModelConfig
from app.classifier.features import extract_features

_clf = None


def _load_classifier():
    global _clf
    if _clf is None:
        if not os.path.exists(settings.CLASSIFIER_MODEL_PATH):
            raise FileNotFoundError(
                f"No trained classifier at {settings.CLASSIFIER_MODEL_PATH}. "
                "Run: python -m app.classifier.train"
            )
        _clf = joblib.load(settings.CLASSIFIER_MODEL_PATH)
    return _clf


def load_routing_config() -> dict:
    with open(settings.ROUTING_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_routing_config(cfg: dict) -> None:
    with open(settings.ROUTING_CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def classify(prompt: str, context: str | None = None) -> tuple[int, float]:
    """Returns (tier: 1|2|3, confidence: 0-1)."""
    clf = _load_classifier()
    feats = extract_features(prompt, context).reshape(1, -1)
    tier = int(clf.predict(feats)[0])
    proba = clf.predict_proba(feats)[0]
    confidence = float(np.max(proba))
    return tier, confidence


def route(
    prompt: str, context: str | None = None, force_tier: str | None = None
) -> tuple[ModelConfig, int, float]:
    """Returns (chosen ModelConfig, tier as int 1/2/3, classifier confidence).
    confidence is 1.0 when force_tier bypasses the classifier."""
    cfg = load_routing_config()

    if force_tier:
        tier = int(force_tier.split("_")[1])
        confidence = 1.0
    else:
        tier, confidence = classify(prompt, context)

    model_name = cfg[f"tier_{tier}"]
    return get_model(model_name), tier, confidence


def verifier_model() -> ModelConfig:
    cfg = load_routing_config()
    return get_model(cfg["verifier_model"])
