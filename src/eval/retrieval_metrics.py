from __future__ import annotations


def precision_at_k(retrieved_ids: list[str], expected_context_ids: list[str], k: int) -> float:
    """Fraction of the top-k retrieved document ids that are relevant (in expected_context_ids).

    A retrieval-only metric - no LLM judge involved, unlike score_example()'s
    faithfulness/context_precision. See internal/mentoring_notes.md (Day 5 / Step 1) for why
    this is a structurally different signal from faithfulness, and why the two must be read
    together, not substituted for each other.

    Plain list[str] in, float out - decoupled from RetrievalResult/Document so this stays
    testable and callable without constructing a retriever, matching score_example()'s
    "plain Python types" seam (Day 2 / Step 4).

    Denominator is len(retrieved_ids[:k]), not k itself - if fewer than k documents were
    retrieved (e.g. a corpus smaller than k), this scores what was actually retrieved rather
    than penalizing the query for a corpus-size limit, matching InMemoryRetriever's own
    top_k-clamped-to-corpus-size behavior (Day 1 / Step 4).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    top_k_ids = retrieved_ids[:k]
    if not top_k_ids:
        return 0.0

    expected = set(expected_context_ids)
    relevant_retrieved = sum(1 for doc_id in top_k_ids if doc_id in expected)
    return relevant_retrieved / len(top_k_ids)
