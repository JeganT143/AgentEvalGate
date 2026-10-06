# Contributing to AgentEvalGate

## Local setup

Requires Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv venv --python 3.13
uv pip install --python .venv/bin/python -r pyproject.toml --extra dev --extra eval --extra api
```

This is the exact install command the CI workflow (`.github/workflows/eval-gate.yml`) runs — if it installs cleanly here, it'll install cleanly in CI.

For anything that calls a real LLM (the `tests/eval/` suite, or running the API/dashboard against real data), copy your OpenAI key into a local `.env`:

```
AEG_API_KEY=sk-...
```

`src/config.py::Settings` is the full, authoritative list of config values (all env-prefixed `AEG_`) — check there rather than guessing what else might be settable.

## Running tests locally

**Unit tests** — fast, free, no API key needed. The LLM call boundary is mocked, so these only verify our own logic (retrieval math, prompt construction, caching, persistence):

```bash
pytest tests/unit/
```

CI runs these first, as their own job, on every pull request. They need no secret, so they run (and must pass) before the paid eval gate starts.

**Eval tests (the gate itself)** — run the real golden dataset through the real pipeline, 51 checks in all:
- 30 `typical` questions graded by a real, pinned LLM judge (RAGAS + DeepEval) against the faithfulness threshold in `src/eval/policy.py`
- 11 out-of-corpus `adversarial` questions that must be declined (`src/eval/refusal.py`, no judge)
- 10 `multi_hop` questions whose search must return every required document (retrieval only, no judge)

Needs `AEG_API_KEY` and takes about 10 minutes (measured: 597.89s for the 30 judged examples; the other checks add seconds). Real cost is metered and reported, about $0.03 per full run. A per-question pass/fail table, with cost, is printed at the end and written to `.cache/gate-report/` (in CI, to the run's summary page and the `gate-report` artifact):

```bash
pytest tests/eval/
```

There's also a `judge-variance.yml` GitHub Actions workflow (manual `workflow_dispatch` only, not run on every PR) that re-measures judge score variance across repeated runs — it costs several hundred real judge calls, so it's not part of the default test loop.

## Running the dashboard locally

```bash
uv pip install --python .venv/bin/python -r pyproject.toml --extra dashboard --extra dashboard-live --extra eval
streamlit run dashboard/app.py
```

Every tab except "Try it live" works with zero config. "Try it live" — which runs a real golden example through the real pipeline and a real RAGAS judge call — needs `AEG_API_KEY` in your `.env`; without it, that tab shows an inline message and the rest of the dashboard is unaffected. The heavy judge stack is only imported when someone presses Run, so it never slows the first page load.

The dashboard is split by concern: `dashboard/app.py` is layout only, `content.py` holds all user-facing copy (write it for someone who has never heard of RAG), `theme.py` the visual identity, `charts.py` the Altair charts, and `verdict.py` the pure, unit-tested pass/block logic for a live run. Streamlit only hot-reloads `app.py`, so restart the server after editing any of the others. `--extra dashboard-live` is only needed for that tab (adds `numpy`/`openai`); `--extra eval` (already installed if you followed the setup above) provides the RAGAS/DeepEval judge.

## Adding a golden example

The golden dataset lives at `data/golden/*.jsonl` — JSONL, one example per line. The field-by-field contract (including *why* each field exists) is documented in [`data/golden/schema.md`](data/golden/schema.md); read that first rather than inferring the shape from an existing row.

In short, each line needs:

- `id` — stable once assigned. Never renumber or reuse an existing id; continue the existing per-category sequence (e.g. the next `typical` id after `typical-030` is `typical-031`).
- `query` — the question to run through the pipeline.
- `expected_context_ids` — ground-truth document ids from the demo corpus (`src/rag/retriever.py::build_demo_corpus`). Verify every id you reference actually exists in the corpus before committing — a typo'd id silently makes an example unscoreable rather than failing loudly.
- `category` — `typical` / `multi_hop` / `adversarial`. The target split across the dataset is roughly 60/20/20; see `schema.md` for why an unbalanced dataset makes the CI gate structurally unable to catch most real regressions. An `adversarial` example with `expected_context_ids: []` is automatically gated on the pipeline declining to answer it, so only use an empty list when no corpus document genuinely answers the question.
- `difficulty` — `easy` / `medium` / `hard`, independent of `category`.
- `reference_answer` — optional; add it for harder examples where spot-checking judge quality matters.

Either append to an existing file (`v1.jsonl` for general-purpose growth) or start a new, purpose-named file (see `adversarial.jsonl` for an example of a focused sub-collection) — both patterns are already in use, pick whichever fits the example you're adding.

A `multi_hop` example is gated on search returning every id in `expected_context_ids`, so list exactly the documents the answer genuinely depends on.

Golden-set changes affect what the merge gate actually tests, not just what it computes — review them with the same rigor as a code change, not less.

## Refreshing the recorded data

The dashboard reads committed measurements from `results/`. Each one is reproducible:

```bash
python -m src.eval.generate_demo_runs --full-gate   # every gated check, both prompts, metered (~$0.06)
python -m src.eval.benchmark_reranker               # reranker off vs on (~$0.003)
python scripts/record_demo_gif.py --url http://localhost:8501   # assets/demo.gif, needs playwright + pillow
```
