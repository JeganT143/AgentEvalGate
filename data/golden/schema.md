# Golden Dataset Schema

The golden dataset is the fixed, curated set of queries the CI eval gate runs
on every PR. It lives at `data/golden/*.jsonl` — one JSON object per line
(JSONL, not a single JSON array), so new examples can be appended without
rewriting the whole file, individual examples diff cleanly in git, and the
dataset can be streamed rather than loaded whole into memory as it grows.

This document defines the schema only. No example data yet — examples are
written against this contract once it's settled, not the other way around.

## Fields

| Field | Type | Required | Purpose |
|---|---|---|---|
| `id` | string | required | Stable identifier for this example. Used to track its score across CI runs over time, so a regression can be attributed to "example `X` got worse," not just "the aggregate dropped." Must not change once assigned, even if the example's text is edited. |
| `query` | string | required | The input question run through the target RAG pipeline. |
| `expected_context_ids` | list[string] | required | Ground-truth document IDs this query should retrieve. Used to compute retrieval precision@k, which separates *retrieval* failures from *generation* failures — a low precision@k with high faithfulness means the model is compensating for weak retrieval, a different root cause than a faithfulness drop with good retrieval. |
| `reference_answer` | string, optional | optional | A human-written reference answer, used to spot-check judge quality on harder examples. Not required for the primary reference-free faithfulness/context-precision metrics, so most examples can omit it — this is what keeps the dataset feasible for a small team to maintain. |
| `category` | enum: `typical` \| `multi_hop` \| `adversarial` | required | Which real-world failure surface this example targets. See "Why `category` exists" below. |
| `difficulty` | enum: `easy` \| `medium` \| `hard` | required | How likely a competent pipeline is to get the example right today. Independent of `category` — see below. |

## `category` definitions

- **`typical`** — a realistic, in-distribution question representative of
  normal usage. The bulk of real traffic; catches basic regressions.
- **`multi_hop`** — requires combining information from more than one
  retrieved document/chunk to answer correctly. Catches regressions where
  retrieval works per-chunk but generation fails to synthesize across
  chunks — a failure mode invisible to single-chunk questions.
- **`adversarial`** — deliberately probes known failure surfaces: questions
  with no answer in the corpus (does the model say "I don't know," or does
  it hallucinate?), ambiguous phrasing, or distractor documents that look
  relevant but aren't.

Target distribution: roughly **60% `typical` / 20% `multi_hop` / 20%
`adversarial`** — skewed toward real traffic composition while reserving
enough of the dataset for the failure modes most likely to be the actual
regression a given PR introduces.

## `difficulty` definitions

- **`easy` / `medium` / `hard`** — independent of `category`: an
  `adversarial` example can be easy to handle correctly, and a `typical`
  example can be surprisingly hard (ambiguous phrasing, genuine nuance).
  Tracking both lets a regression be localized — a regression concentrated
  in previously-*easy* examples is a much stronger signal than one only
  appearing in already-*hard* examples, which may just be expected
  variance.

## Why `category` exists

A merge-blocking gate is only as useful as the hardest failure mode its
dataset can actually surface. If every example were `typical`/`easy`, the
gate would really only be testing "does the pipeline still answer easy,
single-hop, in-distribution questions correctly" — a bar low enough that
almost any regression that matters (broken multi-hop synthesis, new
hallucination on ambiguous or out-of-scope questions, a distractor document
displacing the correct one) can pass through completely untested, because
the gate never asks a question capable of exposing it.

This is the "vanity metric" failure mode: a gate that only ever scores well
on easy inputs produces a green checkmark that means "the pipeline can
still do the easy thing," not "the pipeline is healthy." A gate that passes
without ever having a chance to catch the regression isn't a safety net —
it's a formality that manufactures false confidence, which is arguably
worse than having no gate at all, since a team ships on the strength of a
check that was never capable of catching what's about to reach production.
`category` exists to force the dataset to include the specific failure
surfaces most likely to break silently, so that passing the gate is
evidence of something.
