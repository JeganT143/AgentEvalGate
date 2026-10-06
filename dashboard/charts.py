"""Charts for the dashboard, built with Altair (ships with Streamlit).

Colour follows dashboard/theme.py's palette: ink-muted gray for "before / without",
the accent blue for "after / with" - an emphasis encoding (the "with" value is the
point), never the pass/fail status colours.
"""

from __future__ import annotations

_OFF = "#9297A1"  # --ink-muted
_ON = "#3B5BDB"  # --accent
_RULE = "#C9CDD4"
_INK = "#16181D"
_INK_SECONDARY = "#5B6270"
_GRID = "#E4E6EB"
_FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'


def reranker_dumbbell(by_category: list[dict], labels: dict[str, str], k: int):
    """precision@k per question type, reranker off -> on, one row per category."""
    import altair as alt

    order = [labels.get(row["category"], row["category"]) for row in by_category]
    spans = [
        {
            "type": labels.get(row["category"], row["category"]),
            "off": row["off"],
            "on": row["on"],
            "end": max(row["off"], row["on"]),
            "delta_label": "no change" if row["delta"] == 0 else f"{row['delta']:+.3f}",
            "n": row["n"],
        }
        for row in by_category
    ]
    points = [
        {"type": span["type"], "setting": setting, "value": span[key], "n": span["n"]}
        for span in spans
        for setting, key in (("Reranker off", "off"), ("Reranker on", "on"))
    ]

    y = alt.Y("type:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13, labelColor=_INK, labelPadding=8))
    x_scale = alt.Scale(domain=[0, 1], nice=False)
    x_axis = alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".2f", labelFontSize=11)
    x_title = f"precision@{k}: share of the top {k} search results that are correct"

    rule = (
        alt.Chart(alt.Data(values=spans))
        .mark_rule(strokeWidth=2, color=_RULE)
        .encode(y=y, x=alt.X("off:Q", scale=x_scale, axis=x_axis, title=x_title), x2="on:Q")
    )
    dots = (
        alt.Chart(alt.Data(values=points))
        .mark_circle(size=170, opacity=1, stroke="#FFFFFF", strokeWidth=2)
        .encode(
            y=y,
            x=alt.X("value:Q", scale=x_scale, axis=x_axis, title=x_title),
            color=alt.Color(
                "setting:N",
                scale=alt.Scale(domain=["Reranker off", "Reranker on"], range=[_OFF, _ON]),
                legend=alt.Legend(orient="top", title=None, labelFontSize=12, symbolSize=120),
            ),
            tooltip=[
                alt.Tooltip("type:N", title="Question type"),
                alt.Tooltip("setting:N", title="Setting"),
                alt.Tooltip("value:Q", title=f"precision@{k}", format=".3f"),
                alt.Tooltip("n:Q", title="Questions"),
            ],
        )
    )
    delta = (
        alt.Chart(alt.Data(values=spans))
        .mark_text(align="left", dx=14, fontSize=12, color=_INK_SECONDARY, font=_FONT)
        .encode(y=y, x=alt.X("end:Q", scale=x_scale, axis=x_axis, title=x_title), text="delta_label:N")
    )

    return (
        (rule + dots + delta)
        .properties(height=58 * len(spans) + 20)
        .configure(font=_FONT, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            gridColor=_GRID, domainColor=_GRID, tickColor=_GRID,
            labelColor=_INK_SECONDARY, titleColor=_INK_SECONDARY, titleFontWeight="normal", titleFontSize=12,
        )
        .configure_axisY(grid=False, domain=False, ticks=False)
    )
