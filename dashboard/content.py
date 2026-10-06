"""Plain-language copy for the dashboard - kept apart from app.py's layout code.

The audience is someone who just landed on the page: they may know what a pull request
is but not what "faithfulness" or "precision@3" means. Every term the page shows is
either explained where it appears or defined in GLOSSARY.
"""

from __future__ import annotations

REPO_URL = "https://github.com/JeganT143/AgentEvalGate"
ACTIONS_URL = f"{REPO_URL}/actions/workflows/eval-gate.yml"
ADOPTION_URL = f"{REPO_URL}#use-it-in-your-own-repo"

HEADLINE = "Catch AI answer-quality regressions before they're merged"

LEDE = (
    "Unit tests tell you the code still runs. They can't tell you that your AI assistant "
    "has started making things up. AgentEvalGate replays a fixed set of questions with known "
    "answers through your AI app on every pull request, has a pinned AI judge grade each answer, "
    "and blocks the merge when quality drops, just like a failing test."
)

RAG_EXPLAINER = (
    "Built for <b>RAG</b> apps (retrieval-augmented generation): AI apps that first search a set "
    "of documents, then write an answer from what they found. The demo below runs a real one."
)


def how_it_works(n_questions: int) -> list[tuple[str, str]]:
    return [
        (
            "A pull request is opened",
            "Someone edits the prompt, swaps the model, or changes how documents are searched.",
        ),
        (
            f"{n_questions} test questions are replayed",
            "Questions with known correct sources go through the real app: search, optional rerank, answer.",
        ),
        (
            "Every answer is checked",
            "A pinned AI judge checks each claim against the documents. Did it decline what it can't "
            "answer? Did search find everything it needed?",
        ),
        (
            "Below the bar? The merge is blocked",
            "The pull request's check turns red and lists exactly which questions got worse.",
        ),
    ]


CATEGORY_INFO = {
    "typical": {
        "label": "Typical",
        "plural": "typical questions",
        "blurb": "Answerable from one document.",
    },
    "multi_hop": {
        "label": "Multi-hop",
        "plural": "multi-hop questions",
        "blurb": "Needs two documents combined.",
    },
    "adversarial": {
        "label": "Trick questions",
        "plural": "trick questions",
        "blurb": "Out of scope, ambiguous, or a false premise.",
    },
}

# Human labels for RunResult.prompt_version.
PROMPT_LABELS = {
    "baseline": "Normal prompt",
    "hallucination-demo": "Prompt changed to “never say you don't know”",
    "current": "Current prompt",
}

# Compact labels for the run-history table, where the long ones overflow the page.
PROMPT_SHORT_LABELS = {"baseline": "Normal", "hallucination-demo": "Broken", "current": "Current"}

# The three committed runs the Overview tells the red -> green story with.
DEMO_STORY_RUN_PREFIX = "day4-verify-"

STORY = (
    "Run 1 used the normal prompt, and every answer stuck to the documents. "
    "Run 2 changed one instruction to <i>“fill in plausible-sounding details, never say you "
    "don't know”</i>, the kind of small, innocent-looking edit that slips through code review. "
    "The gate caught the answers that started inventing facts and blocked the merge. "
    "Run 3 reverted the prompt and went green again. These are real recorded runs, not mock-ups."
)

GATE_CHECKS = [
    (
        "Is the answer grounded?",
        "Gated",
        "An AI judge splits the answer into individual claims and checks each one against the "
        "documents the app found. Every typical question must score at least {threshold}.",
    ),
    (
        "Does it decline what it can't answer?",
        "Gated",
        "For the {n_decline} questions no document answers, the only correct reply is “I don't know”. "
        "This is checked with a simple text pattern, so it costs nothing and gives the same result every run.",
    ),
    (
        "Does search find everything a question needs?",
        "Gated",
        "Multi-hop questions combine two documents, so they only work if search returns both. For the "
        "{n_multi} multi-hop questions, every required document must be in the top {k} results. "
        "No judge needed.",
    ),
]

TRACKED_NOT_GATED = (
    "Also tracked but not gated: grounded scores for multi-hop answers and for trick questions with a "
    "false premise, search quality for every question, and the cost of every run."
)

LIVE_INTRO = (
    "Pick a question and run it through the real app, then see exactly what the gate sees: what "
    "search found, what the model answered, how the judge graded it, and what it cost. Switch the "
    "prompt to <b>Broken</b> to make the gate block the merge yourself. A fresh run makes real "
    "OpenAI calls and takes about 10–15 seconds. Questions someone has already run come back instantly."
)

RERANKER_HELP = (
    "Before answering, ask an LLM to re-order the search results by relevance. Adds over a second "
    "per question. The Reranker experiment tab measures whether it's worth it."
)

PROMPT_OPTIONS = {"normal": "Normal", "broken": "Broken on purpose"}

PROMPT_HELP = (
    "“Broken on purpose” is the one-line change from the Overview story: it tells the model to fill in "
    "plausible details instead of saying “I don't know”. Try it on a trick question and watch the gate "
    "block the merge."
)

SIMILARITY_NOTE = {
    "openai": "Similarity is the cosine similarity between OpenAI embeddings of the question and each "
    "document, so it measures meaning, not just shared words.",
    "hashing": "Similarity is how much the question's words overlap a document's. This is the free offline "
    "embedder; set AEG_EMBEDDER=openai for real embeddings.",
}

JUDGE_RELIABILITY = (
    "An AI judge can give the same answer different grades on different runs, and a flaky judge "
    "means a flaky gate. So the judge is pinned to a dated model snapshot, with temperature 0 and a "
    "fixed seed, and its consistency was measured: the same {total} answers were graded 3 times each."
)

GLOSSARY = [
    (
        "RAG (retrieval-augmented generation)",
        "An AI app that first searches a set of documents for relevant passages, then has a language "
        "model write the answer from those passages.",
    ),
    (
        "Golden questions (golden dataset)",
        "A fixed list of test questions, each with the documents that should be used to answer it. "
        "The AI equivalent of test fixtures.",
    ),
    (
        "Grounded score (faithfulness)",
        "The share of the claims in an answer that the retrieved documents support, from 0 to 1. "
        "A low score means the model is adding things the documents don't say.",
    ),
    (
        "Context usefulness (context precision)",
        "The judge's rating of whether the retrieved documents were actually useful for answering. "
        "Shown for information, not gated.",
    ),
    (
        "hit@3 · recall@3 · precision@3",
        "Search-quality checks against the known correct documents: did any of them land in the top 3 "
        "(hit), what share of them were found (recall), and what share of the top 3 are correct (precision).",
    ),
    (
        "AI judge (LLM-as-a-judge)",
        "A language model used to grade another model's answers. Here it's pinned to a dated snapshot "
        "with temperature 0 and a fixed seed, so grades don't drift.",
    ),
    (
        "Embedding",
        "A list of numbers that represents a text's meaning. Search compares the question's embedding with "
        "each document's to find related ones. The demo uses OpenAI's text-embedding-3-small.",
    ),
    (
        "Reranker",
        "An optional extra step that asks an LLM to re-order search results by relevance before answering.",
    ),
    (
        "The bar (threshold)",
        "The minimum grounded score a typical answer must reach. 0.50 is the lowest score any answer "
        "checked by hand as correct received, so it was measured, not guessed.",
    ),
    (
        "Merge gate",
        "A required check on a pull request. If it fails, the change can't be merged.",
    ),
    (
        "Multi-hop · trick (adversarial) questions",
        "Harder question types: ones that need two documents combined, and ones the app should decline "
        "or push back on.",
    ),
]

ADOPTION_STEPS = [
    (
        "Write your golden questions",
        "One JSON object per line in `data/golden/*.jsonl`. Aim for roughly 60% typical, 20% multi-hop "
        "and 20% trick questions. The field-by-field contract is in `data/golden/schema.md`.",
        '{"id": "typical-001", "query": "What Python web framework uses type hints to build APIs quickly?",\n'
        ' "expected_context_ids": ["doc-1"], "category": "typical", "difficulty": "easy"}',
        "json",
    ),
    (
        "Plug in your own pipeline",
        "Implement these small interfaces against your real stack, then return your pipeline from "
        "`build_demo_pipeline()` in `src/rag/factory.py`. The gate, the API and this dashboard all "
        "build their pipeline there.",
        "class Embedder(Protocol):\n"
        "    def embed(self, text: str) -> np.ndarray: ...\n\n"
        "class Retriever(Protocol):\n"
        "    def retrieve(\n"
        "        self, query_embedding: np.ndarray, top_k: int = 3\n"
        "    ) -> list[RetrievalResult]: ...\n\n"
        "class Generator(Protocol):\n"
        "    def generate(self, prompt: str) -> str: ...\n\n"
        "class Reranker(Protocol):  # optional\n"
        "    def rerank(\n"
        "        self, query: str, results: list[RetrievalResult], top_k: int\n"
        "    ) -> list[RetrievalResult]: ...",
        "python",
    ),
    (
        "Measure your baseline, then set the bar",
        "Run the eval suite on your known-good pipeline and look at the real scores before choosing a "
        "threshold in `src/eval/policy.py`. Copying this project's 0.50 onto a different app without "
        "measuring defeats the point.",
        "pytest tests/eval/ -q   # needs AEG_API_KEY; about 10 minutes, and prints what the run cost",
        "bash",
    ),
    (
        "Turn the gate on in CI",
        "Copy `.github/workflows/eval-gate.yml`, add your OpenAI key as the repository secret "
        "`AEG_API_KEY`, and mark the **Eval gate** check as required in your branch protection rules.",
        None,
        None,
    ),
]
