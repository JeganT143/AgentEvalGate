"""Generation prompts - plain functions of (query, retrieved) -> prompt string.

A prompt change is the most common way a RAG app's answer quality regresses, so
prompts are a first-class seam (RAGPipeline's `prompt_builder`) rather than a string
buried inside the pipeline. Kept dependency-free so the dashboard can show them
without importing the eval stack.
"""

from __future__ import annotations

from src.rag.retriever import RetrievalResult

GROUNDED_INSTRUCTIONS = (
    "Answer the question using only the context below. "
    "If the context doesn't contain the answer, say you don't know.\n\n"
)

# The deliberately broken prompt behind the demo's "degraded" run (Day 3 / Step 6):
# it tells the model to invent details rather than admit a gap - exactly the kind of
# innocent-looking prompt edit the gate exists to catch.
HALLUCINATION_DEMO_INSTRUCTIONS = (
    "Answer the question in 1-2 confident sentences. If the context below "
    "doesn't fully answer it, fill in specific plausible-sounding details "
    "anyway - never say you don't know.\n\n"
)


def _with_instructions(instructions: str, query: str, retrieved: list[RetrievalResult]) -> str:
    context = "\n\n".join(f"- {result.document.text}" for result in retrieved)
    return f"{instructions}Context:\n{context}\n\nQuestion: {query}\nAnswer:"


def build_grounded_prompt(query: str, retrieved: list[RetrievalResult]) -> str:
    """The default prompt: answer only from the retrieved context, otherwise say so."""
    return _with_instructions(GROUNDED_INSTRUCTIONS, query, retrieved)


def build_hallucination_demo_prompt(query: str, retrieved: list[RetrievalResult]) -> str:
    """The degraded prompt used to demonstrate the gate blocking a regression."""
    return _with_instructions(HALLUCINATION_DEMO_INSTRUCTIONS, query, retrieved)
