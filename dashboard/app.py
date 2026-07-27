"""Read-only Streamlit dashboard over the eval gate's run results.

Reads exclusively through results_store's functions (Day 4 / Step 1's seam) -
never touches results/*.json directly, so swapping the storage backend later
never requires touching this file. See internal/mentoring_notes.md
(Day 4 / Step 2-3) for why trends over time (not just the latest run), which
categories are failing (not just an aggregate score), and the judge's own
variance are each the point of their panel.

Run locally: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# streamlit run makes the script's own directory sys.path[0], not the repo root
# (unlike pytest, which gets pythonpath=["."] from pyproject.toml) - inserted
# explicitly so `from src...` resolves regardless of invocation directory, which
# matters again once this runs under Dockerfile.dashboard.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.eval.results_store import load_all_run_results, load_judge_variance_summary

st.set_page_config(page_title="AgentEvalGate Dashboard", page_icon="\U0001f4ca", layout="wide")
st.title("AgentEvalGate — Run Results")
st.caption("Read-only view over the eval gate's recorded CI runs.")

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
    st.stop()

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
