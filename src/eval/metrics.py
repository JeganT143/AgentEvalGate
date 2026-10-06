from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI
from ragas.llms.base import llm_factory
from ragas.metrics.collections import ContextPrecisionWithoutReference, Faithfulness

from src.config import get_settings
from src.rag.usage import metered_async_http_client

# Determinism controls, not environment-specific config: these must be identical in every
# environment (local/CI/prod), which is the opposite of what belongs in Settings - see
# internal/mentoring_notes.md (Day 2 / Step 5) for why these are pinned constants here
# rather than env-configurable values.
_JUDGE_TEMPERATURE = 0
_JUDGE_SEED = 42


@dataclass(frozen=True)
class ExampleScores:
    faithfulness: float
    context_precision: float


def score_example(query: str, retrieved_context: list[str], answer: str) -> ExampleScores:
    """Score a single (query, retrieved_context, answer) triple with RAGAS.

    The one seam every eval test depends on instead of calling ragas.metrics.collections
    directly - see internal/mentoring_notes.md (Day 2 / Step 4) for why, and how it keeps
    Day 5's precision@k addition small. Reference-free: no reference_answer required,
    matching this project's reference-free primary-metrics strategy.
    """
    settings = get_settings()
    judge_llm = llm_factory(
        settings.judge_model,
        # Metered like every other OpenAI client (src/rag/usage.py), so judge tokens count
        # toward a run's cost - they're the bulk of it.
        client=AsyncOpenAI(api_key=settings.api_key.get_secret_value(), http_client=metered_async_http_client()),
        temperature=_JUDGE_TEMPERATURE,
        seed=_JUDGE_SEED,
    )

    faithfulness = Faithfulness(llm=judge_llm).score(
        user_input=query, response=answer, retrieved_contexts=retrieved_context
    )
    context_precision = ContextPrecisionWithoutReference(llm=judge_llm).score(
        user_input=query, response=answer, retrieved_contexts=retrieved_context
    )

    return ExampleScores(
        faithfulness=faithfulness.value,
        context_precision=context_precision.value,
    )
