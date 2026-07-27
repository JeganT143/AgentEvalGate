# AgentEvalGate

[![Eval Gate](https://github.com/JeganT143/AgentEvalGate/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/JeganT143/AgentEvalGate/actions/workflows/eval-gate.yml)

An open-source, merge-blocking CI evaluation gate for RAG applications.

**One-line pitch:** Block merges when your RAG faithfulness drops below threshold — the same way a unit-test suite blocks a merge on a broken function.

On every pull request, AgentEvalGate runs a curated golden dataset of queries through the target RAG pipeline, scores the outputs with RAGAS (faithfulness, context precision) via a pinned LLM judge, asserts against data-derived thresholds with DeepEval, and fails the build if quality regresses.
