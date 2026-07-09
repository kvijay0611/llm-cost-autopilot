"""
Phase 5 — Expose the router as an API.

POST /v1/completions      the router picks the model, not the caller
GET  /v1/models           list available models and their costs
GET  /v1/stats            cost savings summary
PUT  /v1/routing-config   update tier -> model mappings without redeploying
GET  /health              liveness check
"""
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.models_registry import MODEL_REGISTRY, most_expensive_model
from app.providers import send_request
from app.routing import route, load_routing_config, save_routing_config
from app.verifier import verify_and_maybe_escalate
from app.schemas import (
    CompletionRequest,
    CompletionResponse,
    RoutingConfigUpdate,
    ModelInfo,
    StatsResponse,
)

app = FastAPI(
    title="LLM Cost Autopilot",
    description="Routes each request to the cheapest model that can handle it, "
    "then verifies quality asynchronously and escalates on failure.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    db.init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/completions", response_model=CompletionResponse)
def create_completion(req: CompletionRequest, background_tasks: BackgroundTasks):
    try:
        model, tier, confidence = route(
            req.prompt, req.context, force_tier=req.force_tier
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    full_prompt = req.prompt if not req.context else f"{req.prompt}\n\nContext:\n{req.context}"
    result = send_request(full_prompt, model)

    baseline_model = most_expensive_model()
    baseline_cost = baseline_model.cost_for(result.input_tokens, result.output_tokens)

    request_id = str(uuid.uuid4())
    db.log_request(
        request_id=request_id,
        prompt=req.prompt,
        complexity_tier=tier,
        classifier_confidence=confidence,
        routed_model=model.name,
        routed_provider=model.provider,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        was_mocked=result.was_mocked,
        baseline_cost_usd=baseline_cost,
    )

    background_tasks.add_task(
        verify_and_maybe_escalate,
        request_id=request_id,
        prompt=full_prompt,
        cheap_output=result.text,
        routed_model_name=model.name,
        tier=tier,
    )

    return CompletionResponse(
        output=result.text,
        routed_model=model.name,
        routed_provider=model.provider,
        complexity_tier=f"tier_{tier}",
        classifier_confidence=confidence,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        request_id=request_id,
        escalation_pending=True,
    )


@app.get("/v1/models", response_model=list[ModelInfo])
def list_models():
    return [
        ModelInfo(
            name=m.name,
            provider=m.provider,
            quality_tier=m.quality_tier,
            cost_per_1m_input=m.cost_per_1m_input,
            cost_per_1m_output=m.cost_per_1m_output,
        )
        for m in MODEL_REGISTRY.values()
    ]


@app.get("/v1/stats", response_model=StatsResponse)
def get_stats():
    return db.fetch_stats()


@app.get("/v1/routing-config")
def get_routing_config():
    return load_routing_config()


@app.put("/v1/routing-config")
def update_routing_config(update: RoutingConfigUpdate):
    cfg = load_routing_config()
    for field in ("tier_1", "tier_2", "tier_3", "verifier_model"):
        val = getattr(update, field)
        if val is not None:
            if val not in MODEL_REGISTRY:
                raise HTTPException(
                    status_code=400, detail=f"Unknown model '{val}' for {field}"
                )
            cfg[field] = val
    save_routing_config(cfg)
    return cfg
