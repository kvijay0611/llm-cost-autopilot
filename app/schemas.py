from pydantic import BaseModel, Field
from typing import Optional, Literal


class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt / task to run.")
    context: Optional[str] = Field(
        None, description="Optional context/document the prompt refers to."
    )
    use_case: Optional[str] = Field(
        None,
        description="Optional hint: 'extraction' | 'summarization' | 'classification' "
        "| 'reasoning' | 'creative'. Improves quality-threshold selection.",
    )
    force_tier: Optional[Literal["tier_1", "tier_2", "tier_3"]] = Field(
        None, description="Bypass the classifier and force a specific tier (debugging)."
    )


class CompletionResponse(BaseModel):
    output: str
    routed_model: str
    routed_provider: str
    complexity_tier: str
    classifier_confidence: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    request_id: str
    escalation_pending: bool  # True if async verification is still queued


class RoutingConfigUpdate(BaseModel):
    tier_1: Optional[str] = None
    tier_2: Optional[str] = None
    tier_3: Optional[str] = None
    verifier_model: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    provider: str
    quality_tier: str
    cost_per_1m_input: float
    cost_per_1m_output: float


class StatsResponse(BaseModel):
    total_requests: int
    total_cost_usd: float
    baseline_cost_usd: float  # cost if every request had gone to the top-tier model
    savings_usd: float
    savings_pct: float
    routing_distribution: dict
    avg_quality_score: Optional[float]
    escalation_rate_pct: float
