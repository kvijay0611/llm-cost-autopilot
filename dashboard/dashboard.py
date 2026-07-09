"""
Phase 4 — Cost and quality dashboard.

Run with: streamlit run dashboard/dashboard.py
Reads directly from the SQLite audit log the API writes to.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px

from app import db
from app.models_registry import MODEL_REGISTRY

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide", page_icon="💸")

st.title("💸 LLM Cost Autopilot — Dashboard")
st.caption(
    "Live view of routing decisions, cost savings, and quality verification "
    "across every request handled by the router."
)

rows = db.fetch_all_rows()

if not rows:
    st.info(
        "No requests logged yet. Start the API (`uvicorn app.main:app`) and send "
        "a few completions, or run `python scripts/load_test.py`, then refresh."
    )
    st.stop()

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

stats = db.fetch_stats()

# --- Headline metrics -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total requests", f"{stats['total_requests']:,}")
c2.metric(
    "Total cost",
    f"${stats['total_cost_usd']:.4f}",
    help="Actual dollars spent using router decisions.",
)
c3.metric(
    "Baseline cost (all top-tier)",
    f"${stats['baseline_cost_usd']:.4f}",
    help="What it would have cost sending every request to the most expensive model.",
)
c4.metric(
    "💰 Cost reduction",
    f"{stats['savings_pct']:.1f}%",
    delta=f"${stats['savings_usd']:.4f} saved",
)

st.divider()

col_left, col_right = st.columns(2)

# --- Routing distribution ----------------------------------------------
with col_left:
    st.subheader("Routing distribution")
    dist = stats["routing_distribution"]
    if dist:
        dist_df = pd.DataFrame(
            {"model": list(dist.keys()), "requests": list(dist.values())}
        )
        fig = px.pie(
            dist_df,
            names="model",
            values="requests",
            hole=0.45,
            color="model",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

# --- Quality score distribution -----------------------------------------
with col_right:
    st.subheader("Quality score distribution")
    quality_df = df.dropna(subset=["quality_score"])
    if not quality_df.empty:
        fig2 = px.histogram(
            quality_df, x="quality_score", nbins=20, range_x=[0, 1]
        )
        fig2.add_vline(
            x=0.55, line_dash="dash", line_color="red",
            annotation_text="escalation threshold"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("Waiting on async verification results...")

st.divider()

# --- Cost over time ------------------------------------------------------
st.subheader("Cost over time: routed vs. baseline")
df_sorted = df.sort_values("timestamp")
df_sorted["cumulative_cost"] = df_sorted["cost_usd"].cumsum()
df_sorted["cumulative_baseline"] = df_sorted["baseline_cost_usd"].cumsum()
melted = df_sorted.melt(
    id_vars="timestamp",
    value_vars=["cumulative_cost", "cumulative_baseline"],
    var_name="series",
    value_name="usd",
)
melted["series"] = melted["series"].map(
    {"cumulative_cost": "Routed (actual)", "cumulative_baseline": "Baseline (all top-tier)"}
)
fig3 = px.line(melted, x="timestamp", y="usd", color="series")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Escalation rate")
    st.metric("% of requests auto-escalated", f"{stats['escalation_rate_pct']:.1f}%")
    st.caption(
        "A request is escalated when the cheap model's output diverges too far "
        "from the verifier model's output — the router then re-runs it at the "
        "higher tier automatically."
    )

with col_b:
    st.subheader("Model registry")
    reg_df = pd.DataFrame(
        [
            {
                "model": m.name,
                "provider": m.provider,
                "quality_tier": m.quality_tier,
                "$/1M in": m.cost_per_1m_input,
                "$/1M out": m.cost_per_1m_output,
            }
            for m in MODEL_REGISTRY.values()
        ]
    )
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Recent requests")
show_cols = [
    "timestamp", "prompt_preview", "complexity_tier", "routed_model",
    "cost_usd", "latency_ms", "quality_score", "escalated", "was_mocked",
]
st.dataframe(
    df[show_cols].sort_values("timestamp", ascending=False).head(100),
    use_container_width=True,
    hide_index=True,
)
