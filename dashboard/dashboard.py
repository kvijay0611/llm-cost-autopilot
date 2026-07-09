"""
Phase 4 — Cost and quality dashboard.

Run with: streamlit run dashboard/dashboard.py
Reads directly from the shared database the API writes to, and can also submit
new live requests to the API itself (via API_BASE_URL) so visitors can test the
router with their own prompts right from this page.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import httpx

from app import db
from app.config import settings
from app.models_registry import MODEL_REGISTRY

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide", page_icon="💸")

db.init_db()

st.title("💸 LLM Cost Autopilot")

with st.expander("ℹ️ What is this?", expanded=True):
    st.markdown(
        """
**LLM Cost Autopilot** is a routing layer that sits in front of multiple LLM
providers. Instead of sending every request to the same expensive model, it:

1. **Classifies** each incoming prompt's complexity (simple / moderate / complex)
   using a trained classifier
2. **Routes** it to the cheapest model that can actually handle that complexity
3. **Verifies** quality asynchronously in the background by comparing the cheap
   model's answer against the top-tier model's answer
4. **Auto-escalates** to the better model if the cheap one's answer looks wrong,
   and logs the failure so the classifier improves over time

Below you can see **example demo traffic** (synthetic prompts sent to
stress-test and showcase the system), and separately **test it yourself** with
your own prompt to see the router make a real decision live.
        """
    )
    st.caption(
        "Note: this deployment runs in mock mode by default (no real LLM calls, "
        "simulated cost/latency) so anyone can try it with zero API cost."
        if settings.FORCE_MOCK_MODE
        else "This deployment is using live LLM calls for at least one provider."
    )

tab_demo, tab_try, tab_all = st.tabs(
    ["📊 Example Demo Run", "🧪 Try It Yourself", "🌐 All Traffic Combined"]
)


def render_stats_block(source: str | None, empty_hint: str):
    stats = db.fetch_stats(source=source)
    rows = db.fetch_all_rows(source=source)

    if stats["total_requests"] == 0:
        st.info(empty_hint)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total requests", f"{stats['total_requests']:,}")
    c2.metric("Total cost", f"${stats['total_cost_usd']:.4f}")
    c3.metric("Baseline cost (all top-tier)", f"${stats['baseline_cost_usd']:.4f}")
    c4.metric(
        "💰 Cost reduction",
        f"{stats['savings_pct']:.1f}%",
        delta=f"${stats['savings_usd']:.4f} saved",
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Routing distribution")
        dist = stats["routing_distribution"]
        if dist:
            dist_df = pd.DataFrame({"model": list(dist.keys()), "requests": list(dist.values())})
            fig = px.pie(dist_df, names="model", values="requests", hole=0.45, color="model")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Quality score distribution")
        df = pd.DataFrame(rows)
        quality_df = df.dropna(subset=["quality_score"]) if not df.empty else df
        if not quality_df.empty:
            fig2 = px.histogram(quality_df, x="quality_score", nbins=20, range_x=[0, 1])
            fig2.add_vline(
                x=settings.ESCALATION_SIMILARITY_THRESHOLD,
                line_dash="dash",
                line_color="red",
                annotation_text="escalation threshold",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("Waiting on async verification results...")

    st.subheader("Recent requests")
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        show_cols = [
            "timestamp", "prompt_preview", "complexity_tier", "routed_model",
            "cost_usd", "latency_ms", "quality_score", "escalated",
        ]
        st.dataframe(
            df[show_cols].sort_values("timestamp", ascending=False).head(50),
            use_container_width=True,
            hide_index=True,
        )


with tab_demo:
    st.caption(
        "Synthetic example traffic — a batch of varied prompts run through the "
        "router to showcase routing distribution and cost savings at scale."
    )
    render_stats_block(
        source="demo",
        empty_hint="No demo data yet. Run `python scripts/load_test.py` "
        "(it tags its traffic as demo data automatically) to populate this view.",
    )

with tab_try:
    st.caption(
        "Send your own prompt through the live router and see exactly how it "
        "gets classified, which model it's routed to, and what it costs."
    )
    with st.form("try_it_form"):
        prompt = st.text_area(
            "Your prompt",
            placeholder="e.g. Summarize this customer complaint in two sentences.",
            height=100,
        )
        context = st.text_area(
            "Optional context",
            placeholder="e.g. the document, ticket, or data your prompt refers to",
            height=80,
        )
        submitted = st.form_submit_button("Route this request →")

    if submitted:
        if not prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            try:
                with st.spinner("Classifying and routing..."):
                    resp = httpx.post(
                        f"{settings.API_BASE_URL}/v1/completions",
                        json={
                            "prompt": prompt,
                            "context": context or None,
                            "source": "live",
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    result = resp.json()

                st.success(
                    f"Routed to **{result['routed_model']}** "
                    f"({result['complexity_tier']}, "
                    f"{result['classifier_confidence']*100:.0f}% classifier confidence)"
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Cost", f"${result['cost_usd']:.6f}")
                m2.metric("Latency", f"{result['latency_ms']} ms")
                m3.metric("Tokens (in/out)", f"{result['input_tokens']}/{result['output_tokens']}")
                st.text_area("Output", result["output"], height=150, disabled=True)
            except Exception as e:
                st.error(
                    f"Couldn't reach the API at {settings.API_BASE_URL}. "
                    f"Error: {e}"
                )

    st.divider()
    st.caption("Live requests you and other visitors have sent:")
    render_stats_block(
        source="live",
        empty_hint="No live requests yet — be the first! Use the form above.",
    )

with tab_all:
    st.caption("Combined view of demo and live traffic together.")
    render_stats_block(source=None, empty_hint="No requests logged yet at all.")

st.divider()
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