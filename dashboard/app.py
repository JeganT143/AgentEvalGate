"""Read-only Streamlit dashboard over the eval gate's run results.

Reads exclusively through results_store.load_all_run_results() (Day 4 / Step 1's
seam) - never touches results/*.json directly, so swapping the storage backend
later never requires touching this file. See internal/mentoring_notes.md
(Day 4 / Step 2) for why trends over time, not just the latest run, are the point.

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

from src.eval.results_store import load_all_run_results

st.set_page_config(page_title="AgentEvalGate Dashboard", page_icon="\U0001f4ca", layout="wide")
st.title("AgentEvalGate — Run Results")
st.caption("Read-only view over the eval gate's recorded CI runs.")

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

st.subheader("Run History")
st.dataframe(
    df[["run_id", "timestamp", "faithfulness", "context_precision", "cost_usd", "passed"]],
    use_container_width=True,
    hide_index=True,
)
