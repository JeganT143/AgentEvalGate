# Block merges when your RAG faithfulness drops below threshold

[![Eval Gate](https://github.com/JeganT143/AgentEvalGate/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/JeganT143/AgentEvalGate/actions/workflows/eval-gate.yml)

The same way a unit-test suite blocks a merge on a broken function — except this grades answer quality, not whether the code compiles. AgentEvalGate is an open-source, merge-blocking CI evaluation gate for RAG applications: it runs a curated golden dataset through your pipeline on every pull request, scores the answers with a pinned LLM judge, and fails the build if quality regresses.

**Demo:** _GIF coming soon — a real PR going red on a deliberately degraded generator prompt, then green after the revert, gated by this repo's own CI check._

**One dashboard, the whole project:** _deploying to Streamlit Community Cloud — link coming soon._ Four tabs in one page:

- **🔍 Try It Live** — pick one of the 58 golden examples, toggle the reranker on/off, and watch a real query run through the real pipeline: retrieved chunks, `precision@k`, the generated answer, and real RAGAS faithfulness/context-precision scores from a real judge call.
- **📊 Eval Gate Results** — the CI gate's run history over time, cost per run, the judge's own measured variance baseline, and which examples are currently failing.
- **🔀 Reranker Impact** — the real, measured before/after numbers for the reranker (precision@k by category, added latency, added cost).
- **🎬 CI Demo** — the red→green GIF above, embedded.

## Quick start: run the dashboard

```bash
docker build -f docker/Dockerfile.dashboard -t agentevalgate-dashboard . && docker run -p 8501:8501 agentevalgate-dashboard
```

Open [http://localhost:8501](http://localhost:8501). Three of the four tabs work immediately with zero configuration. For **🔍 Try It Live** (which makes a real, small-cost OpenAI call per example), pass your key:

```bash
docker run -p 8501:8501 -e AEG_API_KEY=sk-... agentevalgate-dashboard
```

## Quick start: run the API

```bash
docker build -f docker/Dockerfile.api -t agentevalgate-api . && docker run -p 8000:8000 -e AEG_API_KEY=sk-... agentevalgate-api
```

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "What Python web framework uses type hints to build APIs quickly?"}'
```

`/query` is rate-limited (10 requests/minute/IP) and CORS-locked to a configured dashboard origin — see [Configuration](#configuration).

## What it does

On every pull request, AgentEvalGate runs a curated golden dataset of queries through the target RAG pipeline, scores the outputs with RAGAS (faithfulness, context precision) via a pinned LLM judge, computes retrieval `precision@k` against ground-truth document ids, asserts against data-derived thresholds with DeepEval, and fails the build if quality regresses. Everything below is a real, working, separately-verified piece of that system — not a roadmap.

## How it works

```
Query → Embed → Retrieve → Rerank (optional) → Generate → Judge → Gate
```

| Stage | What it is | Where |
|---|---|---|
| Embed | Turns a query into a vector | `Embedder` protocol — `src/rag/pipeline.py` |
| Retrieve | Cosine-similarity nearest-neighbor search over an in-memory corpus | `InMemoryRetriever` — `src/rag/retriever.py` |
| Rerank *(optional)* | Reorders the retrieved candidates with one LLM call, off by default | `LLMReranker` — `src/rag/reranker.py` |
| Generate | Produces the final answer from the (reranked) context | `Generator` protocol — `src/rag/pipeline.py` |
| Judge | Scores the (query, context, answer) triple: faithfulness, context precision, `precision@k` | `src/eval/metrics.py`, `src/eval/retrieval_metrics.py` |
| Gate | Asserts judge scores against a **data-derived** threshold, fails the PR if breached | `.github/workflows/eval-gate.yml`, `tests/eval/test_faithfulness.py` |

**Faithfulness vs. `precision@k` — two different bugs, not two views of the same one:** faithfulness asks *is the answer grounded in what was retrieved*; `precision@k` asks *did retrieval fetch the right documents at all*. Low `precision@k` with high faithfulness means the generator is doing its job correctly on bad context — a retrieval bug. High `precision@k` with low faithfulness means the right documents were fetched and the generator ignored or contradicted them — a generation bug. Conflating the two metrics hides which half of the pipeline actually regressed.

### Repo layout

```
src/
├── rag/          # embed/retrieve/rerank/generate — the pipeline itself
├── eval/         # RAGAS+DeepEval scoring, precision@k, results persistence, caching
├── api/          # FastAPI POST /query — rate-limited, CORS-locked
└── config.py     # the one source of truth for all AEG_* settings
data/golden/      # the versioned JSONL golden dataset + its field-by-field schema
dashboard/        # the 4-tab Streamlit demo UI
tests/
├── unit/         # fast, free, LLM boundary mocked
└── eval/         # real pipeline + real judge, asserted against the threshold
.github/workflows/
├── eval-gate.yml       # the merge-blocking gate — runs on every PR
└── judge-variance.yml  # manual, re-measures judge score variance
docker/           # separate images: API and dashboard
infra/            # Bicep: Azure Container Apps + Key Vault-backed secrets
```

## Configuration

All config is one `pydantic-settings` class (`src/config.py::Settings`), env-prefixed `AEG_`, loaded from a local `.env` or real environment variables — this is the authoritative list, not a copy that can drift from it:

| Variable | Default | Purpose |
|---|---|---|
| `AEG_API_KEY` | *(required)* | OpenAI API key — generation, reranking, and the judge all use it |
| `AEG_MODEL_NAME` | `gpt-4o-mini` | Generation (and reranker) model |
| `AEG_JUDGE_MODEL` | `gpt-4o-mini-2024-07-18` | **Dated snapshot, not a floating alias** — the judge must be pinned so scores don't silently drift when OpenAI updates a model alias |
| `AEG_EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | Cache key namespace for `CachedEmbedder` (today's embedder is a free local placeholder — see [Using this for your own RAG app](#using-this-for-your-own-rag-app)) |
| `AEG_LLM_PROVIDER` | `openai` | Reserved; only `openai` is wired today |
| `AEG_DASHBOARD_ORIGIN` | `http://localhost:8501` | The only browser origin the API's CORS policy allows |
| `AEG_LOG_LEVEL` | `INFO` | Standard Python logging level |

## Using this for your own RAG app

AgentEvalGate is built around three swap points — implement these against your real stack and the pipeline, the eval gate, and the dashboard all keep working unchanged:

```python
class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...

class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...

class Reranker(Protocol):
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]: ...
```

(`src/rag/pipeline.py`, `src/rag/reranker.py`). **Honest limitation:** the retriever (`InMemoryRetriever`, `src/rag/retriever.py`) is a concrete in-memory numpy nearest-neighbor search over a fixed demo corpus, not yet behind its own protocol — swapping in a real vector store (pgvector, Pinecone, etc.) today means writing a class with the same `.retrieve(query_embedding, top_k) -> list[RetrievalResult]` shape and passing it to `RAGPipeline`, which only calls that one method, but there's no formal seam enforcing the contract yet the way `Embedder`/`Generator`/`Reranker` have.

To point the gate at your own project:

1. **Write your golden dataset** — `data/golden/schema.md` is the full field-by-field contract (`id`, `query`, `expected_context_ids`, `category`, `difficulty`, optional `reference_answer`) and *why* each field exists, including why category balance (roughly 60% typical / 20% multi-hop / 20% adversarial) matters for the gate to be worth anything.
2. **Wire your real `Embedder`/`Generator`**, and a retriever over your real corpus, into a `RAGPipeline` — see `src/api/main.py::get_pipeline()` or `dashboard/live_demo.py::get_pipelines()` for the construction pattern.
3. **Measure your own real baseline** before picking a threshold — `tests/eval/test_faithfulness.py`'s `FAITHFULNESS_THRESHOLD` is not a round number; it's the empirical floor of every manually-verified-correct answer this project measured. Copying `0.5` onto a different pipeline without measuring your own baseline first defeats the point.
4. **Copy `.github/workflows/eval-gate.yml`**, point it at your test suite, and wire `AEG_API_KEY` as a repo secret.
5. **Point `dashboard/app.py` at your own `results/`** (it already reads through `src/eval/results_store.py`'s functions, never touching JSON files directly) to get the same 4-tab view over your own data.

## Local development

```bash
uv venv
uv pip install --python .venv/bin/python -r pyproject.toml --extra dev --extra eval --extra api
pytest tests/unit/    # fast, free, no API key needed
pytest tests/eval/    # real pipeline + real judge — needs AEG_API_KEY, costs real (small) money, ~10 minutes for the full typical set
```

Full contributor setup, including running the dashboard locally and adding a golden example, is in [CONTRIBUTING.md](CONTRIBUTING.md). Full history of every build decision and why is in [CHANGELOG.md](CHANGELOG.md).

## Project status

Implemented and verified: the pipeline, the golden dataset (58 examples), the CI gate, retrieval `precision@k`, the reranker (measured before/after), the adversarial test subset, CORS + rate limiting on the API, and the 4-tab dashboard (including a live pipeline call). Not yet done: the demo GIF above, a live Streamlit Community Cloud deployment, and an actual Azure Container Apps deployment of the API (the Bicep in `infra/` is written and reasoned through but not yet compiled/deployed). No `LICENSE` file yet either — added the moment a license is chosen.
