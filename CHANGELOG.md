# Changelog

All notable changes to this project are documented here. Format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This project hasn't cut a tagged/versioned release yet (`pyproject.toml` is still
`0.1.0`), so entries are grouped by build day rather than by version number — real
semantic-version sections will replace this once something actually ships. Dates
below are the real commit dates (`git log --date=short`), not estimates.

## Unreleased — 2026-10-06

### Added
- **Real embeddings** (`src/rag/embedders.py::OpenAIEmbedder`, `text-embedding-3-small`,
  behind `CachedEmbedder`) are now the default. `AEG_EMBEDDER=hashing` keeps the free
  offline placeholder. Measured: hit@3 and recall@3 = 1.00 for every typical and multi-hop
  question (the offline embedder retrieved every required document for only 7/10
  multi-hop questions).
- **Multi-hop search gate** (`tests/eval/test_multi_hop.py`) — every document a multi-hop
  question needs must be in the top 3. Retrieval-only, no judge. Measured before gating:
  10/10, each by a +0.22 to +0.47 cosine margin. Multi-hop *answers* stay tracked rather than
  gated: the grounded prompt declines 6/10 of them, and a prompt that allowed combining
  passages was measured and not adopted (see `src/eval/policy.py`). The gate is now 51 checks.
- **Cost metering** (`src/rag/usage.py`) — an httpx response hook on every OpenAI client
  (generator, reranker, embedder, the RAGAS judge) records real token usage, attributed
  per caller with a ContextVar. Cost appears per question and per run in the gate report,
  on `RunResult.cost_usd`, in the dashboard's history table, and on every live run.
  Measured: a full gate run costs about $0.03.
- **Break-the-prompt switch** in "Try it live" — visitors can run any question with the
  hallucination-demo prompt and watch the gate block it.
- `assets/demo.gif`, recorded from the real dashboard by `scripts/record_demo_gif.py`.
- `python -m src.eval.benchmark_reranker` — a reproducible, metered reranker benchmark
  that replaces the hand-transcribed Day 5 numbers. With real embeddings it measures
  +0.000 precision@3 for +1.4 s per question. The original offline-embedder measurement
  (+0.020) is kept as `results/reranker_benchmark/2026-07-27-offline-embedder.json`.
- `RAGPipeline.retrieve()` — search (+ rerank) without generation.
- **Decline gate** (`tests/eval/test_adversarial.py`, `src/eval/refusal.py`) — the
  11 out-of-corpus `adversarial` questions must be answered with a decline. Pattern-
  based, no judge call. Measured before gating: 11/11 declined on the current prompt,
  0/11 on the hallucination-demo prompt.
- **Per-question gate report** (`src/eval/gate_report.py`, `tests/eval/conftest.py`) —
  written to the GitHub Actions job summary, a `gate-report` artifact, and the terminal.
  Errors (bad key, network) are reported as configuration problems, separately from
  quality failures.
- **Full-gate demo runs** — `python -m src.eval.generate_demo_runs --full-gate` runs
  every gated check for both prompts in parallel and records each as a `RunResult` plus a
  per-question report (`results/gate_reports/`). Recorded: current prompt 51/51 passed,
  broken prompt blocked with 26/51 failing.
- `hit_at_k` / `recall_at_k` retrieval metrics — `precision@3` caps at 0.33 for a
  single-relevant-document question, which read as a failure on perfect retrieval.
- `Retriever` protocol, `src/rag/factory.py::build_demo_pipeline()` (replaces five
  copy-pasted pipeline constructions), `src/rag/prompts.py` and an injectable
  `prompt_builder` (replaces monkey-patching `RAGPipeline._build_prompt`),
  `src/eval/policy.py` (the threshold, shared by the gate and the dashboard).
- `PipelineResult.prompt` and `PipelineResult.timings_ms`.
- API: `GET /health`; `/query` rejects empty queries and caps them at 1,000 characters;
  responses include each retrieved chunk's `id`.
- CI: a free `unit-tests` job that runs first on every PR (unit tests previously never
  ran in CI); a preflight that fails in seconds with a named reason when `AEG_API_KEY`
  is missing, rejected (401) or out of quota (429); Python pinned to 3.13;
  superseded runs cancelled.

### Changed
- **Dashboard redesigned for first-time visitors.** Plain-language headline and a
  4-step "how it works" strip. A new Overview tab leads with the red → green story as
  run cards. "Try it live" is guided: questions grouped by type, the 8-document
  knowledge base shown, the answer displayed in ~2 s while the judge grades, a
  pass/block verdict using the gate's own policy, plain-English score tiles, and the
  exact prompt. Gate history now explains the judge-reliability numbers. The Reranker
  tab leads with a one-sentence answer and a before/after dumbbell chart. "CI Demo"
  (a "coming soon" placeholder) is replaced by "Use it in your repo" with a real gate
  report. A glossary is added, and the mobile layout and table overflow are fixed.
- Dashboard first paint no longer waits on RAGAS/DeepEval imports (lazy-loaded on Run):
  cold first load 2.6 s → 1.3 s locally.
- `measure_variance.py` now covers all 58 golden examples (it silently skipped
  `adversarial.jsonl`).
- README rewritten top-down for newcomers: live demo link, the measured regression
  table, what the gate checks, deployment facts (the live demo runs on Cloud Run).

### Fixed
- `CachedEmbedder` keyed entries by model name only, and the eval gate wrapped the
  offline `HashingEmbedder` under the name `text-embedding-3-small`. That filed 256-dim
  hashing vectors under the real model's name, and they were served to the real embedder
  (a shape mismatch at retrieval time). The key now includes the embedder implementation.
  `CachedEmbedder` is also thread-safe now.
- The deployed dashboard showed Streamlit's default red accent: `Dockerfile.dashboard`
  never copied `.streamlit/config.toml`.
- The live demo inserted the LLM's answer into the page as raw HTML (an injection sink).
  All model output and data are now escaped.
- A failed live call no longer shows the raw exception text (which can include a masked
  key) to the public.

## Day 9 — 2026-07-27

### Changed
- `README.md` significantly expanded: an API quick start alongside the
  existing dashboard one, a "How it works" architecture table (including
  the faithfulness-vs-`precision@k` distinction), a repo-layout tree, a
  full `Configuration` reference for every `AEG_*` setting, a new "Using
  this for your own RAG app" adoption guide (the 3 `Protocol` swap points,
  an honest note on the retriever's current limitation, and a 5-step
  adoption path), and a `Project status` section naming what's done vs.
  not yet (demo GIF, live dashboard/Azure deploys, no `LICENSE` yet).
- Consolidated the two previously-separate gitignored private-notes
  folders (`docs/` and `internal/`) into one (`internal/`) — `build_log.md`
  moved alongside `mentoring_notes.md`; all cross-references between the
  two fixed accordingly. `.gitignore` updated to match.

## Day 8 — 2026-07-27

### Added
- `dashboard/app.py` rebuilt as a 4-tab single page — **🔍 Try It Live** (run
  a real golden example through the real pipeline, toggle the reranker,
  see retrieved chunks/`precision@k`/the answer/real RAGAS scores), the
  original **📊 Eval Gate Results** panels (unchanged, moved into their own
  tab), **🔀 Reranker Impact** (Day 5's measured before/after numbers,
  persisted for the first time as structured data), and **🎬 CI Demo**
  (embeds the Day 7 red/green GIF once it exists).
- `dashboard/live_demo.py` — real pipeline construction and a cached
  (`st.cache_data(persist="disk")`) real-example runner backing the live tab.
- `src/eval/golden_dataset.py::load_golden_examples()` — shared loader for
  all 58 golden examples (`tests/eval/test_faithfulness.py` refactored to
  use it instead of its own inline JSONL parsing).
- `results/reranker_benchmark/summary.json` + `results_store.py::
  load_reranker_benchmark_summary()` — Day 5's reranker measurement,
  persisted as structured data instead of only markdown prose.
- New `dashboard-live` `pyproject.toml` extra (`numpy`, `openai`,
  `pydantic-settings`) for the live tab's dependencies.

### Fixed
- A latent bug in the original dashboard: an early `st.stop()` (when
  `results/` was empty) would have silently blanked every tab after it the
  moment the dashboard went multi-tab, since `st.tabs()` runs every tab body
  in the same script pass. Replaced with per-tab `return`.
- `docker/Dockerfile.dashboard` never copied `data/` — caught by actually
  running the built image, which crashed the live tab with a real
  `FileNotFoundError` until fixed.

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
