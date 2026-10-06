"""The gate's pass/fail policy - one place for the numbers the eval tests enforce and
the dashboard explains, so the live demo can never grade against a different bar than CI.
"""

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

# Which golden categories each check gates. adversarial is gated only on declining
# out-of-corpus questions (src/eval/refusal.py).
FAITHFULNESS_GATED_CATEGORIES = ("typical",)

# multi_hop is gated on RETRIEVAL: every document a question needs must be in the top
# TOP_K. Measured 2026-10-06 before gating, real embeddings (text-embedding-3-small):
# 10/10 multi_hop questions retrieve every required document, each by a wide margin
# (worst required document beats the best document outside the top 3 by +0.22 to +0.47
# cosine similarity - not a flaky boundary). The offline HashingEmbedder retrieves all
# of them for only 7/10, so the gate catches a retrieval regression like that one.
#
# multi_hop ANSWERS stay tracked, not gated: with both documents retrieved, the grounded
# prompt still declines 6/10 of them ("I don't know"). A prompt that allowed combining
# passages answered all 10, but only 6/10 scored >= FAITHFULNESS_THRESHOLD (the answers
# pull in outside facts, e.g. Everest's height, which no document states) and it stopped
# declining an ambiguous adversarial question - so it was measured and not adopted.
RETRIEVAL_GATED_CATEGORIES = ("multi_hop",)
RETRIEVAL_RECALL_BAR = 1.0

# Retrieval depth the demo pipeline answers from - RAGPipeline's default top_k.
TOP_K = 3
