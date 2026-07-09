"""
Phase 4 — Logging. Every request gets a row: timestamp, prompt hash, complexity
tier, routed model, cost, latency, quality score, and whether it was escalated.
"""
import sqlite3
import hashlib
import time
import os
from contextlib import contextmanager

from app.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_preview TEXT NOT NULL,
    complexity_tier INTEGER NOT NULL,
    classifier_confidence REAL NOT NULL,
    routed_model TEXT NOT NULL,
    routed_provider TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    was_mocked INTEGER NOT NULL,
    quality_score REAL,
    escalated INTEGER NOT NULL DEFAULT 0,
    escalated_model TEXT,
    escalated_cost_usd REAL,
    baseline_cost_usd REAL NOT NULL
);
"""


def _ensure_dir():
    d = os.path.dirname(settings.DB_PATH)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@contextmanager
def get_conn():
    _ensure_dir()
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(_SCHEMA)


def log_request(
    request_id: str,
    prompt: str,
    complexity_tier: int,
    classifier_confidence: float,
    routed_model: str,
    routed_provider: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
    was_mocked: bool,
    baseline_cost_usd: float,
):
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO requests
            (request_id, timestamp, prompt_hash, prompt_preview, complexity_tier,
             classifier_confidence, routed_model, routed_provider, input_tokens,
             output_tokens, cost_usd, latency_ms, was_mocked, baseline_cost_usd)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                request_id,
                time.time(),
                prompt_hash,
                prompt[:120],
                complexity_tier,
                classifier_confidence,
                routed_model,
                routed_provider,
                input_tokens,
                output_tokens,
                cost_usd,
                latency_ms,
                int(was_mocked),
                baseline_cost_usd,
            ),
        )


def log_verification(
    request_id: str,
    quality_score: float,
    escalated: bool,
    escalated_model: str | None = None,
    escalated_cost_usd: float | None = None,
):
    with get_conn() as conn:
        conn.execute(
            """UPDATE requests SET quality_score=?, escalated=?, escalated_model=?,
               escalated_cost_usd=? WHERE request_id=?""",
            (
                quality_score,
                int(escalated),
                escalated_model,
                escalated_cost_usd,
                request_id,
            ),
        )


def fetch_stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM requests").fetchone()["c"]
        cost_row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd),0) AS total_cost, "
            "COALESCE(SUM(baseline_cost_usd),0) AS baseline_cost, "
            "AVG(quality_score) AS avg_quality "
            "FROM requests"
        ).fetchone()
        dist_rows = conn.execute(
            "SELECT routed_model, COUNT(*) AS c FROM requests GROUP BY routed_model"
        ).fetchall()
        esc_row = conn.execute(
            "SELECT COALESCE(SUM(escalated),0) AS esc FROM requests"
        ).fetchone()

    distribution = {r["routed_model"]: r["c"] for r in dist_rows}
    total_cost = cost_row["total_cost"] or 0.0
    baseline_cost = cost_row["baseline_cost"] or 0.0
    savings = baseline_cost - total_cost
    savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
    escalation_rate = (esc_row["esc"] / total * 100) if total > 0 else 0.0

    return {
        "total_requests": total,
        "total_cost_usd": round(total_cost, 6),
        "baseline_cost_usd": round(baseline_cost, 6),
        "savings_usd": round(savings, 6),
        "savings_pct": round(savings_pct, 2),
        "routing_distribution": distribution,
        "avg_quality_score": (
            round(cost_row["avg_quality"], 3) if cost_row["avg_quality"] is not None else None
        ),
        "escalation_rate_pct": round(escalation_rate, 2),
    }


def fetch_all_rows() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]
