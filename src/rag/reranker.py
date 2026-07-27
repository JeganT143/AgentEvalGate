from __future__ import annotations

import json
import logging
from typing import Protocol

from openai import OpenAI

from src.rag.retriever import RetrievalResult

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]: ...


class LLMReranker:
    """Reranker seam implementation: reorders retrieved documents by relevance to the
    query with a single structured-output LLM call, then returns the top_k.

    Reuses the already-pinned `openai` client (already authenticated for
    OpenAIGenerator/the RAGAS judge) rather than adding a cross-encoder model - zero
    new dependencies. Full latency/cost-vs-quality tradeoff, the measurement plan, and
    why this was chosen over a cross-encoder: internal/mentoring_notes.md, Day 5 / Step 2.

    Constructed with an explicit client/model_name, not Settings itself - matches
    OpenAIGenerator's pattern of staying a plain seam implementation that doesn't know
    where its config came from. Standalone module, not yet wired into RAGPipeline - see
    the linked notes for why that's a deliberate, separate future step.
    """

    def __init__(self, client: OpenAI, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not results:
            return results

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": self._build_prompt(query, results)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        valid_ids = {result.document.id for result in results}
        ranked_ids = self._parse_ranking(response.choices[0].message.content, valid_ids)
        return self._reorder(results, ranked_ids)[:top_k]

    @staticmethod
    def _build_prompt(query: str, results: list[RetrievalResult]) -> str:
        candidates = "\n\n".join(f"[{result.document.id}] {result.document.text}" for result in results)
        return (
            "Rank the following candidate documents by relevance to the question, most "
            'relevant first. Respond with a JSON object of the exact form {"ranking": '
            '["<doc_id>", ...]} listing every candidate document id exactly once, in '
            "relevance order - no other text.\n\n"
            f"Question: {query}\n\n"
            f"Candidates:\n{candidates}"
        )

    @staticmethod
    def _parse_ranking(raw: str | None, valid_ids: set[str]) -> list[str]:
        # A malformed/hallucinated response is an expected failure mode of LLM structured
        # output, not a code bug - degrade to "no reordering" (empty ranking) rather than
        # raising, so a rerank hiccup doesn't take down the whole pipeline. Mirrors
        # InMemoryRetriever's own top_k-clamping permissiveness rather than OpenAIGenerator's
        # fail-fast style, since this stage is an optional quality enhancement, not a
        # correctness-critical one.
        try:
            ranking = json.loads(raw)["ranking"]
            if not isinstance(ranking, list):
                raise TypeError("'ranking' is not a list")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("LLMReranker: could not parse ranking (%s) - keeping original order", exc)
            return []

        seen: set[str] = set()
        clean_ranking = []
        for doc_id in ranking:
            if doc_id in valid_ids and doc_id not in seen:
                clean_ranking.append(doc_id)
                seen.add(doc_id)
        return clean_ranking

    @staticmethod
    def _reorder(results: list[RetrievalResult], ranked_ids: list[str]) -> list[RetrievalResult]:
        by_id = {result.document.id: result for result in results}
        reordered = [by_id[doc_id] for doc_id in ranked_ids]
        remaining = [result for result in results if result.document.id not in ranked_ids]
        return reordered + remaining
