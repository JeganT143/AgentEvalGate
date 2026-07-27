# Contributing to AgentEvalGate

## Local setup

Requires Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv venv
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

**Eval tests** — run the real golden dataset through the real pipeline and a real, pinned LLM judge (RAGAS + DeepEval), asserted against a data-derived faithfulness threshold. Needs `AEG_API_KEY`, costs real (small) money per run, and takes about 10 minutes for the full `typical` set (measured: 597.89s for 30 examples):

```bash
pytest tests/eval/
```

There's also a `judge-variance.yml` GitHub Actions workflow (manual `workflow_dispatch` only, not run on every PR) that re-measures judge score variance across repeated runs — it costs several hundred real judge calls, so it's not part of the default test loop.

## Adding a golden example

The golden dataset lives at `data/golden/*.jsonl` — JSONL, one example per line. The field-by-field contract (including *why* each field exists) is documented in [`data/golden/schema.md`](data/golden/schema.md); read that first rather than inferring the shape from an existing row.

In short, each line needs:

- `id` — stable once assigned. Never renumber or reuse an existing id; continue the existing per-category sequence (e.g. the next `typical` id after `typical-030` is `typical-031`).
- `query` — the question to run through the pipeline.
- `expected_context_ids` — ground-truth document ids from the demo corpus (`src/rag/retriever.py::build_demo_corpus`). Verify every id you reference actually exists in the corpus before committing — a typo'd id silently makes an example unscoreable rather than failing loudly.
- `category` — `typical` / `multi_hop` / `adversarial`. The target split across the dataset is roughly 60/20/20; see `schema.md` for why an unbalanced dataset makes the CI gate structurally unable to catch most real regressions.
- `difficulty` — `easy` / `medium` / `hard`, independent of `category`.
- `reference_answer` — optional; add it for harder examples where spot-checking judge quality matters.

Either append to an existing file (`v1.jsonl` for general-purpose growth) or start a new, purpose-named file (see `adversarial.jsonl` for an example of a focused sub-collection) — both patterns are already in use, pick whichever fits the example you're adding.

Golden-set changes affect what the merge gate actually tests, not just what it computes — review them with the same rigor as a code change, not less.
