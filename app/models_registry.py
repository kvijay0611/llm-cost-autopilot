"""
Phase 1 — Unified model registry.

Prices are USD per 1M tokens (input, output), matching how providers publish them.
Update these constants whenever provider pricing changes — that's the only place
pricing lives, everything else in the system reads from here.
"""
from dataclasses import dataclass
from typing import Literal, Dict

Provider = Literal["anthropic", "openai", "ollama"]
Quality = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ModelConfig:
    name: str  # internal key used everywhere (registry, routing.yaml, logs)
    provider: Provider
    model_id: str  # the string the provider's API expects
    cost_per_1m_input: float  # USD
    cost_per_1m_output: float  # USD
    avg_latency_ms: int  # rough baseline, used for load-test simulation
    quality_tier: Quality

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.cost_per_1m_input
            + output_tokens / 1_000_000 * self.cost_per_1m_output
        )


# Registry — the five models used for routing in this project.
# Pricing snapshot (edit as needed): Anthropic Aug 2025 list prices, OpenAI list prices,
# Ollama local models are treated as $0 (self-hosted compute cost isn't metered here).
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider="openai",
        model_id="gpt-4o",
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.00,
        avg_latency_ms=1800,
        quality_tier="high",
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        model_id="gpt-4o-mini",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
        avg_latency_ms=700,
        quality_tier="medium",
    ),
    "claude-sonnet": ModelConfig(
        name="claude-sonnet",
        provider="anthropic",
        model_id="claude-sonnet-4-6",
        cost_per_1m_input=3.00,
        cost_per_1m_output=15.00,
        avg_latency_ms=1600,
        quality_tier="high",
    ),
    "claude-haiku": ModelConfig(
        name="claude-haiku",
        provider="anthropic",
        model_id="claude-haiku-4-5",
        cost_per_1m_input=0.25,
        cost_per_1m_output=1.25,
        avg_latency_ms=500,
        quality_tier="low",
    ),
    "llama3-local": ModelConfig(
        name="llama3-local",
        provider="ollama",
        model_id="llama3",
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        avg_latency_ms=2200,  # local CPU/GPU inference, slower but free
        quality_tier="low",
    ),
}


def get_model(name: str) -> ModelConfig:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known models: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def most_expensive_model() -> ModelConfig:
    """Used as the 'what if we sent everything here' baseline for cost-savings math."""
    return max(MODEL_REGISTRY.values(), key=lambda m: m.cost_per_1m_output)
