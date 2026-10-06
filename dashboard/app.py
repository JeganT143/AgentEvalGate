"""AgentEvalGate - the whole project, explained and demonstrated on one page.

Written for someone who just landed here: the header and the Overview tab explain the
problem and the red -> green story in plain language before any metric appears. Then:
a guided live run of the real pipeline + judge, the recorded gate history, the
reranker experiment, and how to adopt it. Every term is explained where it's shown or
in the glossary at the bottom.

Only "Try it live" needs AEG_API_KEY / makes real network calls - everything else is
read-only and free, and the heavy eval stack (RAGAS/DeepEval) is only imported when
someone actually presses Run (see dashboard/live_demo.py).

Copy lives in dashboard/content.py, visual identity in dashboard/theme.py, the live
run's pass/block decision in dashboard/verdict.py - this file is layout only. Data is
read exclusively through results_store's functions (Day 4 / Step 1's seam).

Run locally: streamlit run dashboard/app.py
"""

import html
import math
import sys
from pathlib import Path

# streamlit run makes the script's own directory sys.path[0], not the repo root
# (unlike pytest, which gets pythonpath=["."] from pyproject.toml) - inserted
# explicitly so `from src...`/`from dashboard...` resolve regardless of invocation
# directory, which matters again once this runs under Dockerfile.dashboard.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard import content
from dashboard.theme import (
    apply_theme,
    callout,
    heading,
    pill,
    render_table,
    section_label,
    tile,
    tile_grid,
    verdict_banner,
)
from dashboard.verdict import needs_judge, verdict_for
from src.config import get_settings
from src.eval.golden_dataset import load_golden_examples
from src.eval.policy import FAITHFULNESS_THRESHOLD, RETRIEVAL_GATED_CATEGORIES, TOP_K
from src.eval.refusal import must_decline
from src.eval.results_store import (
    OFFLINE_EMBEDDER_RERANKER_BENCHMARK_PATH,
    RunResult,
    load_all_run_results,
    load_gate_reports,
    load_judge_variance_summary,
    load_reranker_benchmark_summary,
)
from src.rag.prompts import GROUNDED_INSTRUCTIONS, HALLUCINATION_DEMO_INSTRUCTIONS
from src.rag.retriever import build_demo_corpus

st.set_page_config(
    page_title="AgentEvalGate · a CI quality gate for AI answers",
    page_icon="\U0001f9ea",
    layout="wide",
)
apply_theme()

_THRESHOLD = f"{FAITHFULNESS_THRESHOLD:.2f}"


def _fmt_score(value: float | None) -> str:
    # score_example() can legitimately return NaN (RAGAS extracted zero checkable
    # statements) - same guard tests/eval/test_faithfulness.py already uses.
    if value is None:
        return "n/a"
    return "NaN" if isinstance(value, float) and math.isnan(value) else f"{value:.2f}"


def _yes_no(value: float | None) -> str:
    return "n/a" if value is None else ("Yes" if value >= 1.0 else "No")


def _seconds(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def _usd(value: float) -> str:
    # Single runs cost fractions of a cent - show enough digits to be non-zero.
    return f"${value:.4f}" if value >= 0.01 else f"${value:.5f}"


# --- Header -------------------------------------------------------------------------


def render_header(n_questions: int) -> None:
    st.markdown('<div class="aeg-eyebrow">AgentEvalGate · open source</div>', unsafe_allow_html=True)
    st.title(content.HEADLINE)
    st.markdown(f'<div class="aeg-lede">{html.escape(content.LEDE)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="aeg-sub">{content.RAG_EXPLAINER}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aeg-links">'
        f'<a href="{content.REPO_URL}">View the source on GitHub →</a>'
        f'<a href="{content.ADOPTION_URL}">Add it to your own repo →</a>'
        "</div>",
        unsafe_allow_html=True,
    )
    tile_grid(
        [
            tile(title, html.escape(body), kicker=f"Step {i}", kicker_accent=True)
            for i, (title, body) in enumerate(content.how_it_works(n_questions), start=1)
        ]
    )


# --- Overview -----------------------------------------------------------------------


def _run_card(run: RunResult, kicker: str, total_checks: int | None = None) -> str:
    prompt = content.PROMPT_LABELS.get(run.prompt_version, run.prompt_version)
    status = pill("PASSED", True) if run.passed else pill("BLOCKED", False)
    failing = run.failing_examples
    if failing:
        shown = ", ".join(f"<code>{html.escape(f['id'])}</code>" for f in failing[:4])
        more = f" and {len(failing) - 4} more" if len(failing) > 4 else ""
        count = (
            f"{len(failing)} of {total_checks} checks failed"
            if total_checks
            else f"{len(failing)} answer{'s' if len(failing) != 1 else ''} fell below the {_THRESHOLD} bar"
        )
        note = f"{count}: {shown}{more}"
    else:
        note = f"All {total_checks} checks passed." if total_checks else "No answer fell below the bar."
    if "decline_rate" in run.scores:
        note += f"<br>Declined correctly: {run.scores['decline_rate']:.0%} of out-of-scope questions."
    if "retrieval_pass_rate" in run.scores:
        note += f"<br>Multi-hop search found everything: {run.scores['retrieval_pass_rate']:.0%}."
    if run.cost_usd:
        note += f"<br>Cost of the run: {_usd(run.cost_usd)}."
    score = run.scores.get("faithfulness_mean")
    score_html = (
        f'<div class="aeg-run-score">{score:.2f}<span>avg. grounded score</span></div>' if score is not None else ""
    )
    return tile(
        prompt,
        f"{score_html}{note}",
        kicker=kicker,
        extra_class="aeg-run" + ("" if run.passed else " bad"),
        tag=status,
    )


def _hallucinated_decline_quote(report: dict, examples: list[dict]) -> str:
    """A real answer from a recorded gate report where the model should have declined."""
    row = next((r for r in report["rows"] if r["check"] == "declines" and not r["passed"]), None)
    if row is None:
        return ""
    query = next(e["query"] for e in examples if e["id"] == row["example_id"])
    answer = row["detail"].removeprefix("answer: ").strip("'\"").rstrip(".")
    answer = answer if len(answer) <= 110 else answer[:110].rsplit(" ", 1)[0] + "…"
    return (
        f" Asked <i>“{html.escape(query)}”</i>, it replied <i>“{html.escape(answer)}”</i>. "
        "None of the 8 documents says that."
    )


def render_overview_tab(results: list[RunResult], variance: dict | None, examples: list[dict]) -> None:
    story = [r for r in results if r.run_id.startswith(content.DEMO_STORY_RUN_PREFIX)] or results[-3:]
    if story:
        section_label("The 30-second story")
        heading("We broke the prompt on purpose. The gate caught it.")
        tile_grid(
            [_run_card(run, f"Run {i} · {run.timestamp[:10]}") for i, run in enumerate(story, start=1)]
        )
        callout(content.STORY)

    reports = load_gate_reports()
    full = [r for r in results if r.run_id in reports]
    if full:
        heading("Re-run with every check the CI gate runs")
        tile_grid(
            [
                _run_card(r, f"Full gate · {r.timestamp[:10]}", total_checks=len(reports[r.run_id]["rows"]))
                for r in full
            ]
        )
        blocked = next((r for r in full if not r.passed), None)
        if blocked is not None:
            quote = _hallucinated_decline_quote(reports[blocked.run_id], examples)
            callout(
                "With all checks on, the broken prompt fails far more clearly, because it also stopped "
                f"declining questions the documents can't answer.{quote}"
            )

    if story:
        with st.expander("See the one-line prompt change"):
            st.caption("Normal prompt (runs 1 and 3)")
            st.code(GROUNDED_INSTRUCTIONS.strip(), language=None, wrap_lines=True)
            st.caption("Broken prompt (run 2)")
            st.code(HALLUCINATION_DEMO_INSTRUCTIONS.strip(), language=None, wrap_lines=True)

    n_decline = sum(must_decline(e) for e in examples)
    n_multi = sum(e["category"] in RETRIEVAL_GATED_CATEGORIES for e in examples)
    section_label("What the gate checks")
    tile_grid(
        [
            tile(
                title,
                html.escape(body.format(threshold=_THRESHOLD, n_decline=n_decline, n_multi=n_multi, k=TOP_K)),
                tag=f'<span class="aeg-tag{"" if tag == "Gated" else " muted"}">{tag}</span>',
            )
            for title, tag, body in content.GATE_CHECKS
        ]
    )
    st.caption(content.TRACKED_NOT_GATED)

    section_label("Why trust a gate that an AI runs?")
    counts = {cat: sum(e["category"] == cat for e in examples) for cat in content.CATEGORY_INFO}
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Golden questions",
        len(examples),
        help=" · ".join(f"{content.CATEGORY_INFO[c]['label']}: {n}" for c, n in counts.items()),
    )
    if variance is not None:
        col2.metric(
            "Identical re-grades",
            f"{variance['zero_variance_count']}/{variance['total_examples']}",
            help="The same answers were graded 3 times by the pinned judge. This many got the exact "
            "same scores every time. See the Gate history tab for details.",
        )
    col3.metric(
        "The bar",
        _THRESHOLD,
        help="Minimum grounded score for a typical answer: the lowest score any answer checked by hand "
        "as correct received. Measured, not guessed.",
    )
    st.caption("Next: open **Try it live** to run any of these questions yourself.")


# --- Try it live --------------------------------------------------------------------


def render_try_it_live_tab(examples: list[dict]) -> None:
    callout(content.LIVE_INTRO)

    corpus = build_demo_corpus()
    with st.expander(f"What can it answer? The entire knowledge base is {len(corpus)} short documents"):
        render_table([{"id": d.id, "text": d.text} for d in corpus], columns=[("id", "Doc"), ("text", "Text")])
        st.caption("Anything these documents don't cover should get “I don't know”, which is what the trick questions test.")

    try:
        get_settings()
    except Exception:
        st.info(
            "The live demo needs `AEG_API_KEY` configured on the server, so it's switched off here. "
            "Every other tab works without it."
        )
        return

    by_category = {cat: [e for e in examples if e["category"] == cat] for cat in content.CATEGORY_INFO}
    category = st.radio(
        "Question type",
        options=list(content.CATEGORY_INFO),
        format_func=lambda c: f"{content.CATEGORY_INFO[c]['label']} ({len(by_category[c])})",
        captions=[info["blurb"] for info in content.CATEGORY_INFO.values()],
        horizontal=True,
    )
    pool = by_category[category]
    by_id = {e["id"]: e for e in pool}
    example_id = st.selectbox(
        "Question",
        options=[e["id"] for e in pool],
        format_func=lambda eid: by_id[eid]["query"],
        key=f"question-{category}",
    )
    example = by_id[example_id]
    sources = ", ".join(example["expected_context_ids"]) or "none (the documents can't answer this)"
    st.caption(f"{example['id']} · {example['difficulty']} · correct source documents: {sources}")

    col_prompt, col_reranker = st.columns([3, 2])
    prompt = col_prompt.radio(
        "Prompt",
        options=list(content.PROMPT_OPTIONS),
        format_func=content.PROMPT_OPTIONS.get,
        horizontal=True,
        help=content.PROMPT_HELP,
    )
    with col_reranker:
        st.markdown('<div style="height: 1.9rem"></div>', unsafe_allow_html=True)
        reranker_on = st.toggle("Add the AI reranker step", help=content.RERANKER_HELP)

    selection = (example_id, reranker_on, prompt)
    if st.button("Run this question", type="primary"):
        st.session_state["live_run"] = selection
    # Results stay on screen across reruns (e.g. opening an expander) for as long as
    # the selection still matches what was run - both steps are cached, so re-rendering
    # costs nothing.
    if st.session_state.get("live_run") == selection:
        _render_live_result(example, reranker_on, prompt)


def _render_live_result(example: dict, reranker_on: bool, prompt: str) -> None:
    from dashboard.live_demo import judge, run_pipeline

    embedder = get_settings().embedder
    verdict_slot = st.empty()
    try:
        with st.spinner("Step 1 of 2 · Searching the documents and writing an answer…"):
            run = run_pipeline(example["id"], reranker_on, prompt, embedder)
    except Exception as exc:
        st.error(
            f"The live call failed ({type(exc).__name__}). The OpenAI key may be invalid or out of "
            "quota, or the network may be down. The other tabs are unaffected."
        )
        return

    reference = example.get("reference_answer")
    reference_html = (
        f'<div class="aeg-reference"><div class="aeg-qa-label">Reference answer (written by hand)</div>'
        f"{html.escape(reference)}</div>"
        if reference
        else ""
    )
    st.markdown(
        '<div class="aeg-card">'
        f'<div class="aeg-qa-label">Question</div><div class="aeg-question">{html.escape(example["query"])}</div>'
        f'<div class="aeg-qa-label">The app\'s answer</div><div class="aeg-answer">{html.escape(run["answer"])}</div>'
        f"{reference_html}</div>",
        unsafe_allow_html=True,
    )
    scores_box = st.container()

    k = run["k"]
    section_label(f"What the search returned (top {k} of 8 documents)")
    render_table(
        run["retrieved"],
        columns=[
            ("rank", "#"), ("id", "Doc"), ("score", "Similarity"), ("expected", "Correct source?"), ("text", "Text"),
        ],
        card=True,
        float_digits=3,
    )
    precision_note = (
        f" With one correct document, precision@{k} can't go above {1 / k:.2f}."
        if len(example["expected_context_ids"]) == 1
        else ""
    )
    recall = run["recall_at_k"]
    st.caption(
        f"hit@{k}: {_yes_no(run['hit_at_k'])} · recall@{k}: {'n/a' if recall is None else f'{recall:.2f}'} · "
        f"precision@{k}: {run['precision_at_k']:.2f}.{precision_note} {content.SIMILARITY_NOTE[embedder]}"
    )

    judged = None
    if needs_judge(example):
        verdict_slot.markdown(
            verdict_banner("pending", "Grading…", "The AI judge is checking each claim against the documents."),
            unsafe_allow_html=True,
        )
        try:
            with st.spinner("Step 2 of 2 · The AI judge is grading the answer (about 10 seconds)…"):
                judged = judge(example["id"], reranker_on, prompt, embedder)
        except Exception as exc:
            verdict_slot.empty()
            st.error(f"The judge call failed ({type(exc).__name__}). Try again in a moment.")
            return

    verdict = verdict_for(example, run, judged)
    if verdict is not None:
        verdict_slot.markdown(verdict_banner(verdict.kind, verdict.title, verdict.body), unsafe_allow_html=True)

    timings = dict(run["timings_ms"])
    cost = run["cost_usd"]
    if judged is not None:
        timings["judge"] = judged["judge_ms"]
        cost += judged["cost_usd"]
    with scores_box:
        col1, col2, col3, col4, col5 = st.columns(5)
        if judged is not None:
            col1.metric(
                "Grounded score",
                _fmt_score(judged["faithfulness"]),
                help="Faithfulness: the share of the answer's claims that the retrieved documents support "
                f"(0–1). The gate's bar for typical questions is {_THRESHOLD}.",
            )
            col2.metric(
                "Context usefulness",
                _fmt_score(judged["context_precision"]),
                help="Context precision: the judge's rating of whether the retrieved documents were useful "
                "for answering. Shown for information, not gated.",
            )
        else:
            col1.metric(
                "Declined?",
                "Yes" if run["declined"] else "No",
                help="For questions no document answers, the gate checks the reply is a decline "
                "(“I don't know”). No AI judge is needed, because a decline has no claims to check.",
            )
            col2.metric("Grounded score", "n/a", help="Not graded: a decline contains no claims to check.")
        col3.metric(
            "Right doc found?",
            _yes_no(run["hit_at_k"]),
            help=f"hit@{k}: whether any correct source document is in the top {k} search results. "
            "n/a when no document should match.",
        )
        col4.metric(
            "Total time",
            _seconds(sum(timings.values())),
            help=" · ".join(f"{stage}: {_seconds(ms)}" for stage, ms in timings.items()),
        )
        col5.metric(
            "Run cost",
            _usd(cost),
            help="Real OpenAI spend, metered from the usage data on every response: "
            f"search + answer {_usd(run['cost_usd'])}"
            + (f", judge {_usd(judged['cost_usd'])}" if judged is not None else "")
            + ". Cached repeats are free.",
        )

    with st.expander("The exact prompt sent to the model"):
        st.code(run["prompt"], language=None, wrap_lines=True)


# --- Gate history -------------------------------------------------------------------


def _render_gate_report(report: dict, max_rows: int | None, failures_only: bool = False) -> None:
    rows = sorted(report["rows"], key=lambda r: (r["passed"], r["example_id"]))
    if failures_only:
        rows = [r for r in rows if not r["passed"]]
    shown = rows if max_rows is None else rows[:max_rows]
    st.markdown(f"**{html.escape(report['headline'])}**")
    render_table(
        [
            {
                "result": pill("pass", True) if r["passed"] else pill("fail", False),
                "id": r["example_id"],
                "check": "grounded" if r["check"] == "faithfulness" else r["check"],
                "score": r["score"],
                "detail": (r["error"] or r["detail"])[:140],
            }
            for r in shown
        ],
        columns=[("result", "Result"), ("id", "Question"), ("check", "Check"), ("score", "Score"), ("detail", "What happened")],
        card=True,
        html_columns=frozenset({"result"}),
        float_digits=2,
    )
    if len(rows) > len(shown):
        st.caption(f"…and {len(rows) - len(shown)} more rows.")



def render_gate_history_tab(results: list[RunResult], variance: dict | None) -> None:
    callout(
        "Every gate run records the prompt, dataset and judge versions it used, its scores, and which "
        "questions failed. The CI gate also writes a per-question pass/fail table onto the pull "
        "request's check page."
    )

    section_label("Recorded runs (newest first)")
    if not results:
        st.info("No runs recorded yet.")
    else:
        render_table(
            [
                {
                    "result": pill("PASSED", True) if r.passed else pill("BLOCKED", False),
                    "when": r.timestamp.replace("T", " ")[:16],
                    "prompt": content.PROMPT_SHORT_LABELS.get(r.prompt_version, r.prompt_version),
                    "faithfulness": r.scores.get("faithfulness_mean"),
                    "declines": f"{r.scores['decline_rate']:.0%}" if "decline_rate" in r.scores else None,
                    "retrieval": (
                        f"{r.scores['retrieval_pass_rate']:.0%}" if "retrieval_pass_rate" in r.scores else None
                    ),
                    "failing": len(r.failing_examples),
                    "cost": _usd(r.cost_usd) if r.cost_usd else None,
                }
                for r in reversed(results)
            ],
            columns=[
                ("result", "Result"), ("when", "When (UTC)"), ("prompt", "Prompt"),
                ("faithfulness", "Grounded"), ("declines", "Declined"),
                ("retrieval", "Multi-hop"), ("failing", "Failed"), ("cost", "Cost"),
            ],
            card=True,
            html_columns=frozenset({"result"}),
            float_digits=3,
        )
        judges = sorted({r.judge_model for r in results})
        st.caption(
            "Grounded = average grounded score. Declined = share of out-of-scope questions correctly "
            "declined. Multi-hop = share of multi-hop questions whose search found every required "
            "document. Broken = the prompt changed to “never say you don't know”. The July runs graded "
            "5 typical questions for grounding only, before cost was metered (—). The others ran every "
            f"check the CI gate runs, with real metered OpenAI cost, judge included. Judge model: {', '.join(judges)}."
        )
        reports = load_gate_reports()
        blocked = [r for r in results if r.failing_examples]
        if blocked:
            with st.expander("Which questions failed, and why"):
                for r in reversed(blocked):
                    st.caption(f"{r.run_id} · {content.PROMPT_LABELS.get(r.prompt_version, r.prompt_version)}")
                    if r.run_id in reports:
                        _render_gate_report(reports[r.run_id], max_rows=None, failures_only=True)
                    else:
                        render_table(
                            r.failing_examples,
                            columns=[("id", "Question"), ("category", "Type"), ("reason", "Reason")],
                        )

    section_label("Is the AI judge itself reliable?")
    if variance is None:
        st.caption("Not measured yet.")
    else:
        callout(html.escape(content.JUDGE_RELIABILITY.format(total=variance["total_examples"])))
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Identical grades all 3 times",
            f"{variance['zero_variance_count']} of {variance['total_examples']}",
            help="Answers whose grounded score and context usefulness were exactly the same on all 3 re-grades.",
        )
        col2.metric(
            "Run-to-run drift, grounded score",
            f"±{variance['faithfulness_stdev_across_run_means']:.3f}",
            help="Standard deviation of the average grounded score across the 3 grading runs. Small compared "
            f"with the {_THRESHOLD} bar, so the bar isn't sitting inside the noise.",
        )
        col3.metric(
            "Run-to-run drift, context usefulness",
            f"±{variance['context_precision_stdev_across_run_means']:.3f}",
            help="Standard deviation of the average context-usefulness score across the 3 grading runs.",
        )
        with st.expander("The answers whose grades did move"):
            render_table(
                variance["nonzero_variance_examples"],
                columns=[("id", "Question"), ("faithfulness_stdev", "Grounded-score std. dev.")],
            )
            st.caption(
                "Flipped between a number and NaN (no checkable claims) across identical re-grades: "
                + ", ".join(variance["nan_flip_examples"])
            )
            st.caption(f"Method: {variance['source']}")

    section_label("This repository's own gate")
    st.markdown(
        f'<div class="aeg-links"><a href="{content.ACTIONS_URL}">'
        f'<img src="{content.ACTIONS_URL}/badge.svg" alt="Eval Gate status on GitHub Actions"></a>'
        f'<a href="{content.ACTIONS_URL}">See every run on GitHub Actions →</a></div>',
        unsafe_allow_html=True,
    )


# --- Reranker experiment ------------------------------------------------------------


def _reranker_verdict(summary: dict, previous: dict | None) -> str:
    """The tab's one-paragraph answer, written from the numbers rather than fixed copy."""
    p = summary["precision_at_k"]
    k, overall = p["k"], p["all"]
    latency_s = summary["latency_ms_added"]["mean"] / 1000
    per_query = summary["cost_added"]["mean_usd_per_query"]
    labels = {c: info["plural"] for c, info in content.CATEGORY_INFO.items()}
    hit = summary.get("hit_at_k", {}).get("all")

    if overall["delta"] <= 0.005:
        text = "<b>No.</b>" if previous is None else "<b>No, not any more.</b>"
        text += f" Across {overall['n']} questions reranking changed nothing: the share of correct documents in the top {k} stayed at {overall['off']:.3f}"
        if hit and hit["off"] >= 0.999:
            text += f", because search already finds every correct document without it (hit@{k} = {hit['off']:.2f})"
        text += f". It still adds {latency_s:.1f} s and ${per_query:.6f} per question."
        if previous is not None:
            prev = previous["precision_at_k"]["all"]
            text += (
                f" With the earlier offline embedder it did help a little ({prev['delta']:+.3f}). Switching to real "
                "embeddings fixed search at the source and made the reranker redundant."
            )
    else:
        lead = "Only slightly." if overall["delta"] < 0.05 else "Yes."
        text = (
            f"<b>{lead}</b> Across {overall['n']} questions, the share of correct documents in the top {k} went "
            f"from {overall['off']:.3f} to {overall['on']:.3f} ({overall['delta']:+.3f}), while each question "
            f"took {latency_s:.1f} s longer."
        )
        helped = [r for r in p["by_category"] if r["delta"] and r["delta"] > 0]
        if helped:
            text += " It helped " + " and ".join(f"{labels[r['category']]} ({r['delta']:+.3f})" for r in helped) + "."
    return text + " It stays off by default. Measuring a feature before adopting it is the same discipline the gate applies to prompts."


def render_reranker_tab(summary: dict | None, previous: dict | None) -> None:
    if summary is None:
        st.caption("Not measured yet.")
        return

    from dashboard.charts import reranker_dumbbell

    p = summary["precision_at_k"]
    k = p["k"]
    latency = summary["latency_ms_added"]
    cost = summary["cost_added"]
    chart_labels = {c: i["label"] for c, i in content.CATEGORY_INFO.items()}

    heading("Does an AI reranker improve search enough to be worth it?")
    callout(_reranker_verdict(summary, previous))
    st.altair_chart(reranker_dumbbell(p["by_category"], chart_labels, k), use_container_width=True, theme=None)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"precision@{k} gain, all questions", f"{p['all']['delta']:+.3f}")
    col2.metric(
        "Added time per question",
        f"{latency['mean'] / 1000:.1f} s",
        help=f"Mean. Median {latency['median'] / 1000:.1f} s, 95th percentile {latency['p95'] / 1000:.1f} s.",
    )
    col3.metric(
        "Added cost per question",
        f"${cost['mean_usd_per_query']:.6f}",
        help=f"{cost['n_calls']} real reranker calls cost ${cost['total_usd']:.4f} in total (metered).",
    )
    st.caption(
        f"Why is typical stuck at {1 / k:.2f}? Each typical question has exactly one correct document, so even "
        f"perfect search fills only 1 of the {k} slots. That's why the live demo also shows hit@{k}: "
        "“was the right document found at all?”"
    )
    with st.expander(f"Raw numbers and method (measured {summary['measured_at']})"):
        render_table(
            p["by_category"],
            columns=[("category", "Question type"), ("n", "Questions"), ("off", "Off"), ("on", "On"), ("delta", "Change")],
        )
        st.caption(summary["source"])
    if previous is not None:
        with st.expander(f"The earlier measurement, with the offline embedder ({previous['measured_at']})"):
            st.altair_chart(
                reranker_dumbbell(previous["precision_at_k"]["by_category"], chart_labels, k),
                use_container_width=True,
                theme=None,
            )
            st.caption(previous["source"])


# --- Use it in your repo ------------------------------------------------------------


def render_use_it_tab(results: list[RunResult]) -> None:
    callout(
        "AgentEvalGate is built around a few small, swappable interfaces, so the gate, the API and this "
        "dashboard keep working when you point them at your own RAG app. Four steps:"
    )
    for i, (title, body, code, language) in enumerate(content.ADOPTION_STEPS, start=1):
        section_label(f"Step {i}")
        heading(title)
        st.markdown(body)
        if code:
            st.code(code, language=language)

    section_label("What a pull request shows")
    reports = load_gate_reports()
    blocked = next((r for r in reversed(results) if not r.passed and r.run_id in reports), None)
    if blocked is not None:
        st.markdown(
            "Every run adds a per-question table to the pull request's check page, failures first. "
            "Configuration problems (a missing or rejected API key) are reported as such, so they're "
            "never mistaken for a quality drop. This is the real report from running the full gate "
            f"against the broken prompt ({blocked.timestamp[:10]}):"
        )
        _render_gate_report(reports[blocked.run_id], max_rows=8)
    else:
        st.markdown(
            "Every run adds a per-question table to the pull request's check page, failures first. "
            "Configuration problems (a missing or rejected API key) are reported as such, so they're "
            "never mistaken for a quality drop."
        )

    st.markdown(f"Full setup guide: [README → Use it in your own repo]({content.ADOPTION_URL})")


# --- Footer -------------------------------------------------------------------------


def render_footer() -> None:
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("Glossary: the jargon, in plain English"):
        st.markdown(
            '<dl class="aeg-glossary">'
            + "".join(f"<dt>{html.escape(term)}</dt><dd>{html.escape(text)}</dd>" for term, text in content.GLOSSARY)
            + "</dl>",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="aeg-footer">Open source · built with FastAPI, RAGAS, DeepEval and Streamlit · '
        f'<a href="{content.REPO_URL}">source on GitHub</a></div>',
        unsafe_allow_html=True,
    )


examples = load_golden_examples()
results = load_all_run_results()
variance = load_judge_variance_summary()

render_header(len(examples))

tab_overview, tab_live, tab_history, tab_reranker, tab_use = st.tabs(
    ["Overview", "Try it live", "Gate history", "Reranker experiment", "Use it in your repo"]
)
with tab_overview:
    render_overview_tab(results, variance, examples)
with tab_live:
    render_try_it_live_tab(examples)
with tab_history:
    render_gate_history_tab(results, variance)
with tab_reranker:
    render_reranker_tab(
        load_reranker_benchmark_summary(), load_reranker_benchmark_summary(OFFLINE_EMBEDDER_RERANKER_BENCHMARK_PATH)
    )
with tab_use:
    render_use_it_tab(results)

render_footer()
