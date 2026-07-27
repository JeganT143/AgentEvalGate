import math

import numpy as np
import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from src.config import get_settings
from src.eval.cache import CachedEmbedder
from src.eval.golden_dataset import load_golden_examples
from src.eval.metrics import score_example
from src.rag.demo_providers import HashingEmbedder
from src.rag.generators import OpenAIGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import InMemoryRetriever, build_demo_corpus

# Threshold justification (full derivation: internal/build_log.md and
# internal/mentoring_notes.md, Day 2 / Step 6): measuring all 50 golden examples
# through the real pipeline + pinned judge showed `typical` faithfulness tightly
# clustered at 1.0 (28/30) with the only two outliers landing at exactly 0.5 - a
# known RAGAS atomic-statement-decomposition quirk on answers manually verified
# as fully correct and grounded (e.g. "FastAPI uses type hints to help build
# APIs." for a question the context directly answers). 0.5 is therefore the
# empirical floor of today's known-good baseline, not a round number picked in
# the abstract: it's the lowest score any manually-verified-correct typical
# answer has produced, so the gate currently passes the real baseline exactly
# as measured, while still catching anything that scores below what "correct"
# has ever measured as.
#
# Scoped to `typical` only. The same measurement showed `multi_hop` (mean 0.05)
# and `adversarial` (mean ~0.0, one NaN) faithfulness sitting nowhere near this
# range - not because those answers are equally bad, but for two different
# reasons neither of which this threshold (or this metric) is the right tool
# for: multi_hop is a known, tracked retrieval-quality gap (the placeholder
# hashing embedder can't reliably retrieve multiple relevant chunks at once -
# Day 1 / Step 4), and adversarial answers that correctly decline to answer
# ("I don't know") score near-zero faithfulness precisely BECAUSE they contain
# no checkable grounded claims - faithfulness cannot distinguish a correct
# decline from a hallucination for that category. Pooling all three into one
# threshold would either be meaninglessly lenient (anchored low enough to pass
# multi_hop/adversarial) or produce ~20 permanently-red tests that never signal
# a new regression - exactly the "flaky/meaningless gate erodes trust" failure
# mode from Day 2 / Step 5, just from a different cause. multi_hop and
# adversarial need their own metrics/thresholds in a later step, not this one.
FAITHFULNESS_THRESHOLD = 0.5


TYPICAL_EXAMPLES = [row for row in load_golden_examples() if row["category"] == "typical"]


class RagasFaithfulnessMetric(BaseMetric):
    """DeepEval metric wrapping score_example()'s RAGAS faithfulness score.

    score_example() always computes context_precision alongside faithfulness
    (one call, no extra cost) - it's surfaced on self.context_precision and in
    self.reason for visibility, but deliberately NOT gated here: Day 2 / Step 6
    only measured and justified a threshold for faithfulness. Asserting on
    context_precision with an unjustified number would repeat the exact
    "arbitrary round number" mistake that step was written to avoid.

    Runs synchronously only (a_measure delegates to measure) because
    score_example() calls RAGAS's sync .score() wrapper, which explicitly
    raises if invoked from inside a running event loop - see Day 2 / Step 6.
    Callers must use assert_test(..., run_async=False).
    """

    def __init__(self, threshold: float = FAITHFULNESS_THRESHOLD):
        self.threshold = threshold
        self.context_precision: float | None = None

    @property
    def __name__(self) -> str:
        return "RAGAS Faithfulness"

    def measure(self, test_case: LLMTestCase) -> float:
        scores = score_example(
            query=test_case.input,
            retrieved_context=test_case.retrieval_context,
            answer=test_case.actual_output,
        )
        self.score = scores.faithfulness
        self.context_precision = scores.context_precision
        cp_display = (
            "nan" if isinstance(self.context_precision, float) and math.isnan(self.context_precision)
            else f"{self.context_precision:.3f}"
        )
        if isinstance(self.score, float) and math.isnan(self.score):
            # A naive `score >= threshold` silently evaluates to False for NaN with no
            # explanation - Day 2 / Step 5 found this happens for real (RAGAS extracts
            # zero checkable statements from some answers), so it's handled explicitly.
            self.success = False
            self.reason = (
                "faithfulness is NaN: RAGAS extracted zero checkable statements from "
                "the answer, so it cannot be scored - treated as a failure, not "
                f"silently ignored. context_precision={cp_display} (informational, not gated)."
            )
        else:
            self.success = self.score >= self.threshold
            self.reason = (
                f"faithfulness={self.score:.3f} (threshold={self.threshold}); "
                f"context_precision={cp_display} (informational, not gated)"
            )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


@pytest.fixture(scope="module")
def pipeline() -> RAGPipeline:
    settings = get_settings()
    # CachedEmbedder wraps today's free placeholder mainly to exercise/demonstrate the
    # cache mechanism (Day 3 / Step 3-4) - the real payoff lands once a real embeddings
    # client replaces HashingEmbedder, at which point this line is the only thing that
    # needs to change.
    embedder = CachedEmbedder(HashingEmbedder(), model_name=settings.embedding_model_name)
    corpus = build_demo_corpus()
    embeddings = np.stack([embedder.embed(doc.text) for doc in corpus])
    retriever = InMemoryRetriever(documents=corpus, embeddings=embeddings)
    generator = OpenAIGenerator(api_key=settings.api_key.get_secret_value(), model_name=settings.model_name)
    return RAGPipeline(embedder=embedder, retriever=retriever, generator=generator)


@pytest.mark.parametrize("example", TYPICAL_EXAMPLES, ids=[e["id"] for e in TYPICAL_EXAMPLES])
def test_faithfulness(example: dict, pipeline: RAGPipeline) -> None:
    """Runs a `typical` golden example through the real pipeline and real judge,
    asserting RAGAS faithfulness >= FAITHFULNESS_THRESHOLD via DeepEval's assert_test.
    """
    result = pipeline.answer(example["query"])
    test_case = LLMTestCase(
        input=example["query"],
        actual_output=result.answer,
        retrieval_context=[r.document.text for r in result.retrieved_context],
    )
    assert_test(test_case, [RagasFaithfulnessMetric()], run_async=False)
