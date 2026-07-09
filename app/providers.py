"""
Phase 1 — Unified model interface.

send_request(prompt, model_config) -> Response, regardless of which provider
is behind that model. Handles OpenAI, Anthropic, and local Ollama, and falls
back to a deterministic mock so the whole pipeline (routing, verification,
dashboard, load test) can be exercised with zero API keys and zero cost.
"""
import time
import random
import hashlib
from dataclasses import dataclass

import httpx

from app.config import settings
from app.models_registry import ModelConfig


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    model_name: str
    was_mocked: bool


def _estimate_tokens(text: str) -> int:
    """Cheap approximation (~4 chars/token) — good enough for routing/cost math
    without pulling in a tokenizer per provider."""
    return max(1, len(text) // 4)


def _mock_generate(prompt: str, model: ModelConfig) -> str:
    """Deterministic fake output so repeated load tests are reproducible.
    Higher quality tiers produce longer, more 'thorough-looking' text."""
    seed = int(hashlib.sha256((prompt + model.name).encode()).hexdigest(), 16) % (10**8)
    rng = random.Random(seed)
    filler = {
        "low": " It looks correct based on the given input.",
        "medium": " Based on the provided context, here is a structured answer "
        "that addresses the main points raised in the prompt.",
        "high": " After considering multiple angles and the constraints implied by "
        "the prompt, here is a thorough, nuanced answer that weighs trade-offs "
        "and explains the reasoning behind the conclusion.",
    }[model.quality_tier]
    reps = rng.randint(1, 3)
    return f"[mock:{model.name}] Response to: {prompt[:60]}...{filler * reps}"


def _call_anthropic(prompt: str, model: ModelConfig) -> tuple[str, int, int]:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=model.model_id,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _call_openai(prompt: str, model: ModelConfig) -> tuple[str, int, int]:
    with httpx.Client(timeout=30) as client:
        r = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={
                "model": model.model_id,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return (
            text,
            usage.get("prompt_tokens", _estimate_tokens(prompt)),
            usage.get("completion_tokens", _estimate_tokens(text)),
        )


def _call_ollama(prompt: str, model: ModelConfig) -> tuple[str, int, int]:
    with httpx.Client(timeout=60) as client:
        r = client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={"model": model.model_id, "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        data = r.json()
        text = data.get("response", "")
        return text, _estimate_tokens(prompt), _estimate_tokens(text)


def _provider_is_live(model: ModelConfig) -> bool:
    if settings.FORCE_MOCK_MODE:
        return False
    if model.provider == "anthropic":
        return bool(settings.ANTHROPIC_API_KEY)
    if model.provider == "openai":
        return bool(settings.OPENAI_API_KEY)
    if model.provider == "ollama":
        return settings.OLLAMA_ENABLED
    return False


def send_request(prompt: str, model: ModelConfig) -> Response:
    """The single entry point every other module should call. Never raises for
    provider errors in mock-fallback situations — instead degrades to mock so a
    flaky local Ollama server, say, doesn't take down the whole router."""
    start = time.perf_counter()
    live = _provider_is_live(model)
    was_mocked = not live

    if live:
        try:
            if model.provider == "anthropic":
                text, in_tok, out_tok = _call_anthropic(prompt, model)
            elif model.provider == "openai":
                text, in_tok, out_tok = _call_openai(prompt, model)
            elif model.provider == "ollama":
                text, in_tok, out_tok = _call_ollama(prompt, model)
            else:
                raise ValueError(f"Unknown provider {model.provider}")
        except Exception:
            # Live call failed (bad key, rate limit, network) -> degrade to mock
            # rather than crash the request. Portfolio note: in production you'd
            # log this failure distinctly and probably retry/backoff first.
            text = _mock_generate(prompt, model)
            in_tok = _estimate_tokens(prompt)
            out_tok = _estimate_tokens(text)
            was_mocked = True
    else:
        text = _mock_generate(prompt, model)
        in_tok = _estimate_tokens(prompt)
        out_tok = _estimate_tokens(text)
        # simulate realistic latency in mock mode so the dashboard/load test are meaningful
        time.sleep(model.avg_latency_ms / 1000 * random.uniform(0.15, 0.35))

    latency_ms = int((time.perf_counter() - start) * 1000)
    cost = model.cost_for(in_tok, out_tok)

    return Response(
        text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
        cost_usd=cost,
        model_name=model.name,
        was_mocked=was_mocked,
    )
