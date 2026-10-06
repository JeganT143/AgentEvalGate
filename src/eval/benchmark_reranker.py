"""Measure what the LLM reranker buys: retrieval quality, latency and cost, off vs on.

Replaces the hand-transcribed Day 5 numbers with a reproducible measurement that
re-runs against whatever embedder the pipeline is configured with. Generation is
stubbed (StubGenerator) - every metric here is fully determined by retrieve + rerank,
so paying for answers would add cost and no information.

Writes results/reranker_benchmark/summary.json, which the dashboard's
"Reranker experiment" tab reads.

Usage: python -m src.eval.benchmark_reranker
"""

from __future__ import annotations

import json
import statistics
import time

from src.config import get_settings
from src.eval.golden_dataset import load_golden_examples
from src.eval.results_store import DEFAULT_RERANKER_BENCHMARK_PATH
from src.eval.retrieval_metrics import hit_at_k, precision_at_k, recall_at_k
from src.rag.demo_providers import StubGenerator
from src.rag.factory import build_demo_pipeline
from src.rag.usage import track_usage

CATEGORIES = ("typical", "multi_hop", "adversarial")


def _summarise(rows: list[dict], metric: str) -> dict:
    def block(subset: list[dict]) -> dict:
        scored = [r for r in subset if r[f"{metric}_off"] is not None]
        if not scored:
            return {"n": 0, "off": None, "on": None, "delta": None}
        off = statistics.mean(r[f"{metric}_off"] for r in scored)
        on = statistics.mean(r[f"{metric}_on"] for r in scored)
        return {"n": len(scored), "off": round(off, 4), "on": round(on, 4), "delta": round(on - off, 4)}

    return {
        "by_category": [{"category": c, **block([r for r in rows if r["category"] == c])} for c in CATEGORIES],
        "all": block(rows),
    }


def main() -> None:
    settings = get_settings()
    off = build_demo_pipeline(settings, generator=StubGenerator())
    on = build_demo_pipeline(settings, generator=StubGenerator(), reranker=True)
    k = off.top_k

    rows, latencies = [], []
    with track_usage() as rerank_meter:
        for example in load_golden_examples():
            expected = example["expected_context_ids"]
            # Off first: it also warms the embedding cache, so the metered "on" pass below
            # pays for reranking only.
            with track_usage():
                ids_off = [r.document.id for r in off.retrieve(example["query"])]
            result_on = on.answer(example["query"])
            ids_on = [r.document.id for r in result_on.retrieved_context]
            latencies.append(result_on.timings_ms["rerank"])
            row = {"id": example["id"], "category": example["category"]}
            for name, fn in (("precision", precision_at_k), ("hit", hit_at_k), ("recall", recall_at_k)):
                row[f"{name}_off"], row[f"{name}_on"] = fn(ids_off, expected, k), fn(ids_on, expected, k)
            rows.append(row)
            print(f"  {example['id']}: off={ids_off} on={ids_on}")

    latencies.sort()
    summary = {
        "measured_at": time.strftime("%Y-%m-%d", time.gmtime()),
        "source": (
            f"python -m src.eval.benchmark_reranker: all {len(rows)} golden examples, reranker OFF vs ON, "
            f"embedder={settings.embedder} ({settings.embedding_model_name if settings.embedder == 'openai' else 'offline'}), "
            f"real {settings.model_name} reranker calls, StubGenerator for generation (retrieval metrics are "
            "fully determined by retrieve + rerank). Cost metered from OpenAI's usage data (src/rag/usage.py)."
        ),
        "embedder": settings.embedder,
        "precision_at_k": {"k": k, **_summarise(rows, "precision")},
        "hit_at_k": {"k": k, **_summarise(rows, "hit")},
        "recall_at_k": {"k": k, **_summarise(rows, "recall")},
        "latency_ms_added": {
            "mean": round(statistics.mean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "p95": round(latencies[int(0.95 * (len(latencies) - 1))], 1),
            "min": round(latencies[0], 1),
            "max": round(latencies[-1], 1),
        },
        "cost_added": {
            "n_calls": rerank_meter.calls,
            "prompt_tokens": rerank_meter.prompt_tokens,
            "completion_tokens": rerank_meter.completion_tokens,
            "total_usd": round(rerank_meter.cost_usd, 6),
            "mean_usd_per_query": round(rerank_meter.cost_usd / len(rows), 6),
        },
    }
    DEFAULT_RERANKER_BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_RERANKER_BENCHMARK_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("precision_at_k", "latency_ms_added", "cost_added")}, indent=2))


if __name__ == "__main__":
    main()
