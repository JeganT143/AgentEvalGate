# Changelog

All notable changes to this project are documented here. Format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This project hasn't cut a tagged/versioned release yet (`pyproject.toml` is still
`0.1.0`), so entries are grouped by build day rather than by version number — real
semantic-version sections will replace this once something actually ships. Dates
below are the real commit dates (`git log --date=short`), not estimates.

## Day 5 — 2026-07-27

### Added
- `src/eval/retrieval_metrics.py::precision_at_k` — deterministic retrieval-quality
  metric scored against `expected_context_ids`, independent of any LLM judge.
- `src/rag/reranker.py::LLMReranker` — LLM-based reranker (reuses the existing
  `openai` client, no new dependencies), wired into `RAGPipeline` as an optional,
  off-by-default stage.
- `data/golden/adversarial.jsonl` — 8 new adversarial examples covering
  out-of-corpus questions, ambiguous/zero-antecedent phrasing, and
  conflicting-context traps.
- `CONTRIBUTING.md`, `CHANGELOG.md`.

### Changed
- `README.md` rewritten: H1 is now the one-line impact statement, plus a demo-GIF
  placeholder, a dashboard-link placeholder, and a verified one-command Docker
  quick start.

## Day 4 — 2026-07-27

### Added
- `src/eval/results_store.py` — `RunResult` schema and append-only JSON
  persistence for eval run history.
- `dashboard/app.py` — Streamlit dashboard (faithfulness/context-precision trend
  chart, cost-per-run chart, run-history table).
- Dashboard panels: judge-variance baseline, currently-failing-examples-by-category.
- `docker/Dockerfile.dashboard` — separate, slim image for the dashboard only.
- `dashboard/requirements.txt` for Streamlit Community Cloud deployment.
- Real recorded run history (3 real pipeline + judge runs) so the dashboard shows
  actual data instead of an empty state.

### Fixed
- Corrected a previously-reported judge-variance zero-variance count (48/50 → the
  actual 47/50 — a denominator mix-up in the original analysis).
- Restructured `pyproject.toml`'s dependency extras so the dashboard image no
  longer pulls in API-only dependencies (`fastapi`, `openai`, etc.).

## Day 3 — 2026-07-27

### Added
- `.github/workflows/eval-gate.yml` — the merge-blocking CI eval gate itself.
- `src/eval/cache.py::CachedEmbedder` — content-hash-keyed embedding cache, with
  hit/miss logging proven against real repeated runs.
- `.github/workflows/judge-variance.yml` — on-demand (`workflow_dispatch`) CI
  judge-variance re-measurement.
- `README.md` with a live CI status badge.

### Changed
- `eval-gate.yml` now also triggers on `push` to `master`, not just `pull_request`,
  so the badge has a status to render outside of open PRs.

### Security
- `AEG_API_KEY` wired via GitHub Actions repo secrets rather than committed —
  withheld from fork-triggered `pull_request` runs by default.

## Day 2 — 2026-07-25

### Added
- 45 more golden examples (`data/golden/v1.jsonl`) — 50 total across the dataset,
  matching the agreed 60/20/20 `typical`/`multi_hop`/`adversarial` split.
- `ragas` + `deepeval` as the eval-harness dependencies.
- `src/eval/metrics.py::score_example()` — wraps RAGAS faithfulness and context
  precision behind one seam.
- Judge model/temperature/seed pinned for reproducibility; real score variance
  measured across 3 runs of all 50 golden examples.
- `tests/eval/test_faithfulness.py` — a data-derived faithfulness threshold
  (`0.5`), scoped to the `typical` category.

### Fixed
- A tokenizer bug in the placeholder embedder that silently broke retrieval for
  any question-shaped query (punctuation wasn't stripped before hashing).

## Day 1 — 2026-07-24

### Added
- Initial repo skeleton (`src/`, `tests/`, `data/`, `docs/`,
  `.github/workflows/`, `docker/`).
- Pinned core dependencies (`fastapi`, `uvicorn`, `pydantic-settings`, `pytest`)
  in `pyproject.toml`.
- `src/config.py::Settings` — centralized, 12-factor config.
- `src/rag/retriever.py::InMemoryRetriever` — in-memory numpy nearest-neighbor
  search over a small demo corpus.
- `src/rag/pipeline.py::RAGPipeline` — the embed → retrieve → prompt → generate
  orchestrator.
- `POST /query` FastAPI endpoint, wired via dependency injection.
- Unit tests for the retriever and pipeline (LLM call boundary mocked).
- `data/golden/schema.md` — the golden dataset's field-by-field contract.
- `data/golden/v0.jsonl` — the first 5 golden examples, across all 3 categories.
- A real `OpenAIGenerator`, wired into `/query` in place of the placeholder stub.
