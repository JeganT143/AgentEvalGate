# Block merges when your RAG faithfulness drops below threshold

[![Eval Gate](https://github.com/JeganT143/AgentEvalGate/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/JeganT143/AgentEvalGate/actions/workflows/eval-gate.yml)

The same way a unit-test suite blocks a merge on a broken function — except this grades answer quality, not whether the code compiles. AgentEvalGate is an open-source, merge-blocking CI evaluation gate for RAG applications.

**Demo:** _GIF coming Day 7 — a real PR going red on a deliberately degraded generator prompt, then green after the revert, gated by this repo's own CI check._

**Live dashboard:** _deploying to Streamlit Community Cloud — link coming soon._

## Quick start: run the dashboard

```bash
docker build -f docker/Dockerfile.dashboard -t agentevalgate-dashboard . && docker run -p 8501:8501 agentevalgate-dashboard
```

Open [http://localhost:8501](http://localhost:8501).

## What it does

On every pull request, AgentEvalGate runs a curated golden dataset of queries through the target RAG pipeline, scores the outputs with RAGAS (faithfulness, context precision) via a pinned LLM judge, asserts against data-derived thresholds with DeepEval, and fails the build if quality regresses.
