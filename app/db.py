"""
Phase 4 — Logging. Every request gets a row: timestamp, prompt hash, complexity
tier, routed model, cost, latency, quality score, and whether it was escalated.

Supports two backends, chosen automatically:
  - SQLite (default): zero setup, perfect for local dev and single-process demos.
  - Postgres: used automatically when DATABASE_URL is set (e.g. on Render), so
    multiple services (API + dashboard) can share one database over the network
    instead of each having its own isolated local file.
"""
import hashlib
import os
import time
from contextlib import contextmanager

from app.config import settings

USE_POSTGRES = bool(settings.DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras
else:
    import sqlite3

_PH = "%s" if USE_POSTGRES else "?"  # SQL placeholder style differs by backend

_SCHEMA_SQLITE = """
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
    baseline_cost_usd REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'live'
);
"""

_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    timestamp DOUBLE PRECISION NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_preview TEXT NOT NULL,
    complexity_tier INTEGER NOT NULL,
    classifier_confidence DOUBLE PRECISION NOT NULL,
    routed_model TEXT NOT NULL,
    routed_provider TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd DOUBLE PRECISION NOT NULL,
    latency_ms INTEGER NOT NULL,
    was_mocked INTEGER NOT NULL,
    quality_score DOUBLE PRECISION,
    escalated INTEGER NOT NULL DEFAULT 0,
    escalated_model TEXT,
    escalated_cost_usd DOUBLE PRECISION,
    baseline_cost_usd DOUBLE PRECISION NOT NULL,
    source TEXT NOT NULL DEFAULT 'live'
);
"""


def _ensure_dir():
    if USE_POSTGRES:
        return
    d = os.path.dirname(settings.DB_PATH)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@contextmanager
def get_conn():
    _ensure_dir()
    if USE_POSTGRES:
        conn = psycopg2.connect(
            settings.DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor
        )
    else:
        conn = sqlite3.connect(settings.DB_PATH)
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_SCHEMA_POSTGRES if USE_POSTGRES else _SCHEMA_SQLITE)
        cur.close()
    # Migration: tables created before the 'source' column existed need it
    # added explicitly (CREATE TABLE IF NOT EXISTS above only handles brand
    # new tables). Safe to run every startup — errors on an already-present
    # column are swallowed.
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            if USE_POSTGRES:
                cur.execute(
                    "ALTER TABLE requests ADD COLUMN IF NOT EXISTS "
                    "source TEXT NOT NULL DEFAULT 'live'"
                )
            else:
                cur.execute(
                    "ALTER TABLE requests ADD COLUMN source TEXT NOT NULL DEFAULT 'live'"
                )
            cur.close()
    except Exception:
        pass  # column already exists (SQLite has no IF NOT EXISTS for ALTER)


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
    source: str = "live",
):
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""INSERT INTO requests
            (request_id, timestamp, prompt_hash, prompt_preview, complexity_tier,
             classifier_confidence, routed_model, routed_provider, input_tokens,
             output_tokens, cost_usd, latency_ms, was_mocked, baseline_cost_usd, source)
            VALUES ({','.join([_PH] * 15)})""",
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
                source,
            ),
        )
        cur.close()


def log_verification(
    request_id: str,
    quality_score: float,
    escalated: bool,
    escalated_model: str | None = None,
    escalated_cost_usd: float | None = None,
):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""UPDATE requests SET quality_score={_PH}, escalated={_PH},
               escalated_model={_PH}, escalated_cost_usd={_PH} WHERE request_id={_PH}""",
            (
                quality_score,
                int(escalated),
                escalated_model,
                escalated_cost_usd,
                request_id,
            ),
        )
        cur.close()


def fetch_stats(source: str | None = None) -> dict:
    """source=None returns stats across everything; pass 'demo' or 'live' to
    scope the numbers to just that bucket."""
    where = f" WHERE source={_PH}" if source else ""
    params = (source,) if source else ()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) AS c FROM requests{where}", params)
        total = cur.fetchone()["c"]

        cur.execute(
            "SELECT COALESCE(SUM(cost_usd),0) AS total_cost, "
            "COALESCE(SUM(baseline_cost_usd),0) AS baseline_cost, "
            "AVG(quality_score) AS avg_quality "
            f"FROM requests{where}",
            params,
        )
        cost_row = cur.fetchone()

        cur.execute(
            f"SELECT routed_model, COUNT(*) AS c FROM requests{where} GROUP BY routed_model",
            params,
        )
        dist_rows = cur.fetchall()

        cur.execute(f"SELECT COALESCE(SUM(escalated),0) AS esc FROM requests{where}", params)
        esc_row = cur.fetchone()
        cur.close()

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


def fetch_all_rows(source: str | None = None) -> list[dict]:
    where = f" WHERE source={_PH}" if source else ""
    params = (source,) if source else ()
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM requests{where} ORDER BY timestamp DESC", params)
        rows = cur.fetchall()
        cur.close()
    return [dict(r) for r in rows]