"""Visual identity for the dashboard - one shared stylesheet + a small table
renderer, kept separate from app.py's rendering logic (Day 10 / Step 1).

Palette/type choices are deliberate, not Streamlit defaults - see
internal/mentoring_notes.md, Day 10 / Step 1, for the design reasoning.
Status colors (--good/--bad) are reserved for pass/fail semantics only, never
reused as a general accent - the same rule this project's own eval gate is
built around (a faithfulness score is either past or below threshold, nothing
in between gets to borrow that color).
"""

from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
:root {
  --surface: #FAFAFA;
  --surface-raised: #FFFFFF;
  --ink: #16181D;
  --ink-secondary: #5B6270;
  --ink-muted: #9297A1;
  --accent: #3B5BDB;
  --accent-soft: #EEF1FD;
  --border: #E4E6EB;
  --good: #1C7C4D;
  --good-soft: #E8F5EC;
  --bad: #C1292E;
  --bad-soft: #FCEAEA;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}

/* Strip Streamlit's own chrome - a portfolio piece shouldn't announce its framework */
#MainMenu, footer, header[data-testid="stHeader"], div[data-testid="stToolbar"],
div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {
  visibility: hidden;
  height: 0;
}

html, body, [class*="css"] { font-family: var(--font-sans); }
.stApp { background: var(--surface); color: var(--ink); }
.block-container { padding-top: 2.25rem; max-width: 1024px; }

/* Header block */
.aeg-eyebrow {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  color: var(--ink-muted);
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}
h1 { font-weight: 700; letter-spacing: -0.01em; color: var(--ink); }
.aeg-lede { color: var(--ink-secondary); font-size: 1rem; line-height: 1.55; max-width: 62ch; }
.aeg-links { margin-top: 0.9rem; margin-bottom: 0.4rem; }
.aeg-links a { color: var(--accent); text-decoration: none; font-size: 0.9rem; font-weight: 500; }
.aeg-links a:hover { text-decoration: underline; }
.aeg-links img { vertical-align: middle; }

hr, div[data-testid="stMarkdownContainer"] hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.5rem 0;
}

/* Tabs - plain text, underline indicator, no pill background */
div[data-testid="stTabs"] button[role="tab"] {
  font-family: var(--font-sans);
  font-weight: 600;
  font-size: 0.92rem;
  color: var(--ink-muted);
  padding: 0.5rem 0.1rem;
  margin-right: 1.75rem;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--ink);
}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  background-color: var(--accent);
  height: 2px;
}
div[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: var(--border); }

/* Section labels (used instead of raw st.subheader for a quieter, consistent look) */
.aeg-section-label {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-muted);
  margin: 1.6rem 0 0.6rem 0;
}

/* Metrics - mono numerals, quiet uppercase labels, a card frame */
div[data-testid="stMetric"] {
  background: var(--surface-raised);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.9rem 1.1rem 0.7rem 1.1rem;
}
label[data-testid="stMetricLabel"] p {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-muted);
  /* Streamlit's own default truncates a long label to one ellipsized line -
     overridden to wrap instead, since silently clipping a real measured
     label's text is the wrong tradeoff for this project (Day 10 / Step 1). */
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: unset !important;
}
div[data-testid="stMetricValue"] p {
  font-family: var(--font-mono);
  color: var(--ink);
  font-size: 1.55rem;
}

/* Cards for grouped content (retrieved chunks, answer, etc.) */
.aeg-card {
  background: var(--surface-raised);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.1rem 1.25rem;
  margin-bottom: 1rem;
}

/* Custom tables (st.dataframe's canvas grid can't be CSS-styled at the cell
   level - small, known-shape tables render as plain HTML instead, see
   render_table() below) */
.aeg-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.aeg-table th {
  text-align: left;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-muted);
  border-bottom: 1px solid var(--border);
  padding: 0.5rem 0.75rem;
}
.aeg-table td {
  padding: 0.55rem 0.75rem;
  border-bottom: 1px solid var(--border);
  color: var(--ink);
}
.aeg-table tr:last-child td { border-bottom: none; }
.aeg-table td.num, .aeg-table th.num { font-family: var(--font-mono); text-align: right; }
.aeg-table td.mono { font-family: var(--font-mono); }
.aeg-check-yes { color: var(--good); font-weight: 700; }
.aeg-check-no { color: var(--ink-muted); }

/* Buttons */
.stButton button {
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 6px;
  font-weight: 600;
  padding: 0.5rem 1.25rem;
}
.stButton button:hover { background: #2F49B8; color: white; }

/* Status pill, used for pass/fail-flavored callouts */
.aeg-pill {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 0.78rem;
  font-weight: 600;
  padding: 0.15rem 0.6rem;
  border-radius: 999px;
}
.aeg-pill-good { background: var(--good-soft); color: var(--good); }
.aeg-pill-bad { background: var(--bad-soft); color: var(--bad); }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="aeg-section-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def render_table(rows: list[dict], columns: list[tuple[str, str]], card: bool = False) -> None:
    """Render a small, known-shape table as plain styled HTML.

    `st.dataframe` renders through a canvas-based grid (glide-data-grid) in
    modern Streamlit, which CSS can't reach cell-by-cell - these tables are
    all small and shape-known ahead of time, so plain HTML gives full control
    (mono/right-aligned numbers, a check glyph for booleans) at the cost of
    losing st.dataframe's built-in sort/scroll, an acceptable trade at this
    scale (see internal/mentoring_notes.md, Day 10 / Step 1).

    `columns` is a list of (key, display_label) pairs, in column order. Bools
    render as a check glyph (accent-colored yes / muted em-dash for no);
    floats/ints render right-aligned in the mono face; everything else as
    plain escaped text.

    The whole table is built as ONE html string and passed to a single
    st.markdown call - never split a `<div>` open/close across two separate
    st.markdown calls, each one renders into its own isolated container, so
    an "open" tag from one call can never be closed by a later one; the
    optional `card` wrapper is applied here, in the same string, for exactly
    that reason.
    """
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, bool):
                glyph = '<span class="aeg-check-yes">&#10003;</span>' if value else '<span class="aeg-check-no">&#8212;</span>'
                cells.append(f"<td>{glyph}</td>")
            elif isinstance(value, float):
                cells.append(f'<td class="num">{value:.4f}</td>')
            elif isinstance(value, int):
                cells.append(f'<td class="num">{value}</td>')
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    table_html = f'<table class="aeg-table"><thead><tr>{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'
    if card:
        table_html = f'<div class="aeg-card">{table_html}</div>'
    st.markdown(table_html, unsafe_allow_html=True)
