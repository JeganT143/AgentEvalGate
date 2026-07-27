"""AgentEvalGate — single-page demo UI, the whole project in one place.

Four tabs: a live pipeline call against a real golden example, the eval gate's
CI run history, the reranker's measured before/after impact, and the CI-gate
red/green demo. Only "Try It Live" needs AEG_API_KEY / makes real network
calls - the other three stay read-only and free, matching the original
single-tab dashboard's zero-config contract (Day 4).

Reads exclusively through results_store's functions (Day 4 / Step 1's seam) -
never touches results/*.json directly, so swapping the storage backend later
never requires touching this file. See internal/mentoring_notes.md
(Day 4 / Step 2-3, Day 8 / Step 2) for why trends over time, category
failures, judge variance, reranker impact, and a live pipeline call are each
the point of their own panel.

Run locally: streamlit run dashboard/app.py
"""

import math
import sys
from pathlib import Path

# streamlit run makes the script's own directory sys.path[0], not the repo root
# (unlike pytest, which gets pythonpath=["."] from pyproject.toml) - inserted
# explicitly so `from src...`/`from dashboard...` resolve regardless of invocation
# directory, which matters again once this runs under Dockerfile.dashboard.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from dashboard.live_demo import run_example
from src.config import get_settings
from src.eval.golden_dataset import load_golden_examples
from src.eval.results_store import (
    load_all_run_results,
    load_judge_variance_summary,
    load_reranker_benchmark_summary,
)

st.set_page_config(page_title="AgentEvalGate", page_icon="\U0001f4ca", layout="wide")

# Reading order for the "Try It Live" dropdown - not alphabetical (which would
# read adversarial, multi_hop, typical), but the same typical -> multi_hop ->
# adversarial order the golden dataset schema itself presents them in.
_CATEGORY_ORDER = {"typical": 0, "multi_hop": 1, "adversarial": 2}


def _fmt_score(value: float) -> str:
    # score_example() can legitimately return NaN (RAGAS extracted zero checkable
    # statements) - same guard tests/eval/test_faithfulness.py already uses.
    return "NaN" if isinstance(value, float) and math.isnan(value) else f"{value:.3f}"


def render_header() -> None:
    st.title("Block merges when your RAG faithfulness drops below threshold")
    st.markdown(
        "[![Eval Gate](https://github.com/JeganT143/AgentEvalGate/actions/workflows/eval-gate.yml/badge.svg)]"
        "(https://github.com/JeganT143/AgentEvalGate/actions/workflows/eval-gate.yml) &nbsp; "
        "[GitHub repo](https://github.com/JeganT143/AgentEvalGate)"
    )
    st.caption(
        "AgentEvalGate, end to end, in one page: a live pipeline call, the eval "
        "gate's CI history, the reranker's measured impact, and the CI gate "
        "catching a real regression."
    )
    st.markdown("**The pipeline: Retrieve → Rerank (optional) → Generate → Judge → Gate**")


def render_try_it_live_tab() -> None:
    try:
        get_settings()
    except Exception:
        st.info("Live demo requires `AEG_API_KEY` to be configured — showing static results only.")
        return

    examples = sorted(load_golden_examples(), key=lambda e: (_CATEGORY_ORDER[e["category"]], e["id"]))
    selected_id = st.selectbox(
        "Golden example",
        options=[e["id"] for e in examples],
        format_func=lambda eid: next(
            f"[{e['category']}] {e['id']} — {e['query']}" for e in examples if e["id"] == eid
        ),
    )
    reranker_on = st.toggle("Reranker on")

    if st.button("Run"):
        with st.spinner("Calling the real pipeline + real RAGAS judge..."):
            try:
                result = run_example(selected_id, reranker_on)
            except Exception as exc:
                st.error(f"Live pipeline call failed: {exc}")
                return

        st.subheader("Retrieved chunks")
        st.dataframe(
            pd.DataFrame(result["retrieved"])[["id", "score", "expected", "text"]],
            use_container_width=True,
            hide_index=True,
        )
        st.metric("precision@3", f"{result['precision_at_k']:.3f}")

        st.subheader("Generated answer")
        st.write(result["answer"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Faithfulness", _fmt_score(result["faithfulness"]))
        col2.metric("Context precision", _fmt_score(result["context_precision"]))
        col3.metric("Latency", f"{result['latency_ms']:.0f} ms")


def render_eval_gate_results_tab() -> None:
    # Independent of run history - a fresh deploy with zero CI runs yet still shows
    # the judge's own measured reliability baseline.
    st.subheader("Judge Variance Baseline")
    variance = load_judge_variance_summary()
    if variance is None:
        st.caption("Not measured yet.")
    else:
        vcol1, vcol2, vcol3 = st.columns(3)
        vcol1.metric(
            "Examples with zero variance",
            f"{variance['zero_variance_count']}/{variance['total_examples']}",
        )
        vcol2.metric("Faithfulness stdev (across runs)", f"{variance['faithfulness_stdev_across_run_means']:.4f}")
        vcol3.metric(
            "Context precision stdev (across runs)",
            f"{variance['context_precision_stdev_across_run_means']:.4f}",
        )
        with st.expander("Non-zero-variance and NaN-flip examples"):
            st.write("Graded variance:", variance["nonzero_variance_examples"])
            st.write("NaN-flip (numeric ↔ NaN across identical repeats):", variance["nan_flip_examples"])
            st.caption(variance["source"])

    results = load_all_run_results()

    if not results:
        st.info(
            "No run results yet. Once the eval gate is wired to append results, "
            "each CI run will show up here."
        )
        return

    df = pd.DataFrame(
        [
            {
                "run_id": r.run_id,
                "timestamp": pd.to_datetime(r.timestamp),
                "faithfulness": r.scores.get("faithfulness_mean"),
                "context_precision": r.scores.get("context_precision_mean"),
                "cost_usd": r.cost_usd,
                "passed": r.passed,
            }
            for r in results
        ]
    ).sort_values("timestamp")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total runs", len(df))
    col2.metric("Latest faithfulness", f"{df['faithfulness'].iloc[-1]:.3f}")
    col3.metric("Latest run", "✅ passed" if df["passed"].iloc[-1] else "❌ failed")

    st.subheader("Faithfulness & Context Precision Over Time")
    st.line_chart(df.set_index("timestamp")[["faithfulness", "context_precision"]])

    st.subheader("Cost per Run")
    st.bar_chart(df.set_index("run_id")[["cost_usd"]])

    st.subheader("Currently Failing Examples")
    latest_run = results[-1]
    if latest_run.failing_examples:
        st.dataframe(pd.DataFrame(latest_run.failing_examples), use_container_width=True, hide_index=True)
    else:
        st.success("No failing examples in the latest run.")

    st.subheader("Run History")
    st.dataframe(
        df[["run_id", "timestamp", "faithfulness", "context_precision", "cost_usd", "passed"]],
        use_container_width=True,
        hide_index=True,
    )


def render_reranker_impact_tab() -> None:
    summary = load_reranker_benchmark_summary()
    if summary is None:
        st.caption("Not measured yet.")
        return

    st.dataframe(
        pd.DataFrame(summary["precision_at_k"]["by_category"]),
        use_container_width=True,
        hide_index=True,
    )
    st.metric("All categories: precision@3 delta", f"+{summary['precision_at_k']['all']['delta']:.4f}")

    col1, col2 = st.columns(2)
    col1.metric("Added latency (mean)", f"{summary['latency_ms_added']['mean']:.1f} ms")
    col2.metric("Added cost (total, 50 real calls)", f"${summary['cost_added']['total_usd']:.4f}")
    st.caption(summary["source"])


def render_ci_demo_tab() -> None:
    gif_path = Path("assets/demo.gif")
    if gif_path.exists():
        st.image(
            str(gif_path),
            caption="A real PR going red on a degraded generator prompt, then "
            "green after the revert — gated by this repo's own CI check.",
        )
    else:
        st.info("Demo GIF coming soon — see the open `demo/degrade-faithfulness` PR.")


render_header()

tab_live, tab_results, tab_reranker, tab_ci = st.tabs(
    ["🔍 Try It Live", "📊 Eval Gate Results", "🔀 Reranker Impact", "🎬 CI Demo"]
)

with tab_live:
    render_try_it_live_tab()
with tab_results:
    render_eval_gate_results_tab()
with tab_reranker:
    render_reranker_impact_tab()
with tab_ci:
    render_ci_demo_tab()
