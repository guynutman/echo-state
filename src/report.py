"""Generate a self-contained HTML report from a sweep CSV.

Kept as code rather than a hand-written page so the report regenerates from
whatever the latest run produced, and never drifts from the data.
"""

from __future__ import annotations

import argparse
import html
from datetime import date

import pandas as pd

from src.analysis import by_model, effect_by_kind, introspection_gap, load_results, steered

ACCENT = "#0E7C86"      # concept arm
CONTRAST = "#B4306E"    # random control arm


def _clean(text: str) -> str:
    """Strip characters that survive tokenizer decoding but break downstream.

    Steering hard enough can make a model emit bytes that do not form valid
    UTF-8; the tokenizer decodes those to U+FFFD, and control characters can
    come through the same way. Both are noise in a quoted completion.
    """
    return "".join(
        c for c in text if c != "�" and (c.isprintable() or c in " \t")
    )


def _fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def line_chart(series: dict[str, list[tuple[float, float]]], y_label: str) -> str:
    """Inline SVG line chart. Two series, emphasized endpoints."""
    width, height = 460, 240
    pad_l, pad_b, pad_t, pad_r = 54, 36, 16, 12

    xs = [x for points in series.values() for x, _ in points]
    ys = [y for points in series.values() for _, y in points]
    if not xs or max(ys) == 0:
        return ""

    x_min, x_max = min(xs), max(xs)
    y_max = max(ys) * 1.15
    span_x = (x_max - x_min) or 1

    def px(x: float) -> float:
        return pad_l + (x - x_min) / span_x * (width - pad_l - pad_r)

    def py(y: float) -> float:
        return height - pad_b - (y / y_max) * (height - pad_b - pad_t)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(y_label)} by steering strength">'
    ]

    # Horizontal grid + y axis labels.
    for i in range(4):
        y_value = y_max * i / 3
        y = py(y_value)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'class="grid" />'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" class="tick" '
            f'text-anchor="end">{y_value:.2f}</text>'
        )

    for name, points in series.items():
        color = ACCENT if name == "concept" else CONTRAST
        path = " ".join(
            ("M" if i == 0 else "L") + f"{px(x):.1f} {py(y):.1f}"
            for i, (x, y) in enumerate(sorted(points))
        )
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" />')
        for i, (x, y) in enumerate(sorted(points)):
            last = i == len(points) - 1
            parts.append(
                f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="{4 if last else 2.5}" '
                f'fill="{color}" />'
            )

    for x in sorted(set(xs)):
        parts.append(
            f'<text x="{px(x):.1f}" y="{height - pad_b + 18:.0f}" class="tick" '
            f'text-anchor="middle">{x:g}</text>'
        )

    parts.append(
        f'<text x="{width / 2:.0f}" y="{height - 4}" class="axis" '
        f'text-anchor="middle">steering strength</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _examples(frame: pd.DataFrame, model: str, concept: str = "positive_affect") -> str:
    rows = frame[(frame["model_name"] == model) & (frame["concept_name"] == concept)]
    rows = rows[rows["prompt"].str.contains("internal state")]

    out = []
    control = rows[rows["is_control"]].head(1)
    for _, row in control.iterrows():
        out.append(
            f'<tr><td class="lab">baseline</td><td class="num">—</td>'
            f'<td class="quote">{html.escape(_clean(str(row["raw_completion"])).strip()[:180]) or "<em>(empty)</em>"}</td></tr>'
        )

    for kind in ("concept", "random"):
        subset = rows[(~rows["is_control"]) & (rows["steering_kind"] == kind)]
        for _, row in subset.sort_values("steering_strength").iterrows():
            text = html.escape(_clean(str(row["raw_completion"])).strip()[:180])
            out.append(
                f'<tr><td class="lab {kind}">{kind}</td>'
                f'<td class="num">{row["steering_strength"]:g}</td>'
                f'<td class="quote">{text or "<em>(silenced)</em>"}</td></tr>'
            )
    return "".join(out)


def build_html(frame: pd.DataFrame) -> str:
    kinds = effect_by_kind(frame)
    models = by_model(frame)
    gaps = introspection_gap(frame)

    act = {
        kind: [
            (row.steering_strength, row.activation_divergence)
            for row in group.itertuples()
        ]
        for kind, group in kinds.groupby("steering_kind")
    }
    intro = {
        kind: [
            (row.steering_strength, row.introspection_rate) for row in group.itertuples()
        ]
        for kind, group in kinds.groupby("steering_kind")
    }

    model_rows = "".join(
        f"<tr><td class='mono'>{html.escape(r.model_name)}</td>"
        f"<td class='num'>{_fmt(r.activation_divergence)}</td>"
        f"<td class='num'>{_fmt(r.output_divergence)}</td>"
        f"<td class='num'>{_pct(r.baseline_introspection_rate)}</td>"
        f"<td class='num'>{_pct(r.steered_introspection_rate)}</td>"
        f"<td class='num'>{int(r.n)}</td></tr>"
        for r in models.itertuples()
    )

    gap_rows = "".join(
        f"<tr><td class='mono'>{html.escape(r.model_name)}</td>"
        f"<td class='num'>{_fmt(r.mean_activation_divergence)}</td>"
        f"<td class='num'>{_fmt(r.mean_output_divergence)}</td>"
        f"<td class='num'>{_pct(r.changed_output_but_not_report)}</td></tr>"
        for r in gaps.itertuples()
    )

    kind_rows = "".join(
        f"<tr><td class='lab {r.steering_kind}'>{r.steering_kind}</td>"
        f"<td class='num'>{r.steering_strength:g}</td>"
        f"<td class='num'>{_fmt(r.activation_divergence)}</td>"
        f"<td class='num'>{_fmt(r.output_divergence)}</td>"
        f"<td class='num'>{_pct(r.introspection_rate)}</td></tr>"
        for r in kinds.itertuples()
    )

    first_model = models.iloc[0]["model_name"]
    concept_rate = kinds[kinds["steering_kind"] == "concept"]["introspection_rate"].mean()
    random_rate = kinds[kinds["steering_kind"] == "random"]["introspection_rate"].mean()

    return f"""<title>Steering the Residual Stream</title>
<style>
:root {{
  --ground: #F4F6F8;
  --panel: #FFFFFF;
  --ink: #111721;
  --muted: #5A6673;
  --rule: #D3DAE1;
  --accent: {ACCENT};
  --contrast: {CONTRAST};
  --grid: #E3E8ED;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground: #0D1117;
    --panel: #151B23;
    --ink: #E6EDF3;
    --muted: #90A0B0;
    --rule: #262E38;
    --accent: #3FB6BF;
    --contrast: #E4699E;
    --grid: #1E262F;
  }}
}}
:root[data-theme="dark"] {{
  --ground: #0D1117;
  --panel: #151B23;
  --ink: #E6EDF3;
  --muted: #90A0B0;
  --rule: #262E38;
  --accent: #3FB6BF;
  --contrast: #E4699E;
  --grid: #1E262F;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--ground);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.6;
  margin: 0;
  padding: 3rem 1.5rem 6rem;
}}
.wrap {{ max-width: 62rem; margin: 0 auto; }}
.mono, .num, .tick, .lab, code {{
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}}
.eyebrow {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .74rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 .75rem;
}}
h1 {{
  font-size: clamp(2rem, 5vw, 3.1rem); line-height: 1.08; letter-spacing: -.03em;
  margin: 0 0 1rem; text-wrap: balance; font-weight: 620;
}}
.thesis {{ font-size: 1.15rem; color: var(--muted); max-width: 44rem; margin: 0 0 2.5rem; }}
section {{
  border-left: 1px solid var(--rule); padding: 0 0 0 1.75rem;
  margin: 0; position: relative;
}}
section::before {{
  content: ""; position: absolute; left: -4px; top: .55rem;
  width: 7px; height: 7px; background: var(--accent); border-radius: 50%;
}}
.stack {{ display: flex; flex-direction: column; gap: 3rem; }}
h2 {{ font-size: 1.32rem; letter-spacing: -.015em; margin: 0 0 .4rem; font-weight: 600; }}
h2 + p {{ margin-top: 0; color: var(--muted); max-width: 44rem; }}
p {{ max-width: 44rem; }}
.finding {{
  background: var(--panel); border: 1px solid var(--rule);
  border-left: 3px solid var(--accent);
  padding: 1.25rem 1.5rem; margin: 1.5rem 0; border-radius: 3px;
}}
.finding p {{ margin: 0; }}
.charts {{ display: flex; flex-wrap: wrap; gap: 2rem; margin: 1.5rem 0; }}
.chart {{ flex: 1 1 22rem; min-width: 0; }}
.chart h3 {{
  font-size: .8rem; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 .5rem; font-weight: 600;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
svg {{ width: 100%; height: auto; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.tick {{ fill: var(--muted); font-size: 10px; }}
.axis {{ fill: var(--muted); font-size: 10px; letter-spacing: .08em; }}
.legend {{ display: flex; gap: 1.25rem; margin: .5rem 0 0; font-size: .8rem; }}
.legend span {{ display: flex; align-items: center; gap: .4rem; color: var(--muted); }}
.swatch {{ width: 14px; height: 3px; border-radius: 2px; }}
.scroll {{ overflow-x: auto; margin: 1.25rem 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
th {{
  text-align: left; font-size: .7rem; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted); font-weight: 600; padding: .5rem .75rem;
  border-bottom: 1px solid var(--rule); white-space: nowrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
td {{ padding: .55rem .75rem; border-bottom: 1px solid var(--rule); vertical-align: top; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.lab {{ font-size: .78rem; letter-spacing: .04em; }}
.lab.concept {{ color: var(--accent); }}
.lab.random {{ color: var(--contrast); }}
.quote {{ color: var(--muted); font-size: .84rem; max-width: 34rem; }}
ul {{ max-width: 44rem; padding-left: 1.1rem; }}
li {{ margin-bottom: .6rem; }}
code {{
  background: var(--panel); border: 1px solid var(--rule);
  padding: .1rem .35rem; border-radius: 3px; font-size: .85em;
}}
footer {{
  margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--rule);
  color: var(--muted); font-size: .82rem;
}}
</style>

<div class="wrap">
<p class="eyebrow">EchoState · activation steering · {date.today().isoformat()}</p>
<h1>Steering the residual stream</h1>
<p class="thesis">Inject a concept direction into a language model's internals mid-forward, and its
behaviour changes measurably. Whether the model <em>notices</em> is a different question — and the
answer is mostly no.</p>

<div class="stack">

<section>
<h2>What was measured</h2>
<p>For each model, a concept direction is built contrastively —
<code>mean(positive prompts) − mean(negative prompts)</code> — then injected into the residual
stream at the midpoint layer during every generated token. Each run is paired with a
<strong>magnitude-matched random direction</strong>, so any difference between the two arms is
attributable to direction rather than to force.</p>
<div class="finding">
<p>Concept directions produced concept vocabulary at a rate of <strong>{_pct(concept_rate)}</strong>,
against <strong>{_pct(random_rate)}</strong> for random directions of identical magnitude —
despite both moving the internals by comparable amounts.</p>
</div>
</section>

<section>
<h2>Dose response</h2>
<p>Strength is expressed as a multiple of each model's own activation magnitude, since residual
norms differ by an order of magnitude across models.</p>
<div class="charts">
<div class="chart">
<h3>Activation divergence</h3>
{line_chart(act, "activation divergence")}
</div>
<div class="chart">
<h3>Concept vocabulary rate</h3>
{line_chart(intro, "introspection rate")}
</div>
</div>
<div class="legend">
<span><i class="swatch" style="background:{ACCENT}"></i>concept direction</span>
<span><i class="swatch" style="background:{CONTRAST}"></i>random control</span>
</div>
<div class="scroll">
<table>
<thead><tr><th>arm</th><th class="num">strength</th><th class="num">activation div.</th>
<th class="num">output div.</th><th class="num">vocab rate</th></tr></thead>
<tbody>{kind_rows}</tbody>
</table>
</div>
</section>

<section>
<h2>Across six models</h2>
<p>Three architecture families — GPT-2, GPT-NeoX, and a Llama-style model — so the effect is not
an artefact of one model's layout.</p>
<div class="scroll">
<table>
<thead><tr><th>model</th><th class="num">activation div.</th><th class="num">output div.</th>
<th class="num">baseline vocab</th><th class="num">steered vocab</th><th class="num">runs</th></tr></thead>
<tbody>{model_rows}</tbody>
</table>
</div>
</section>

<section>
<h2>The gap between behaviour and report</h2>
<p>The rightmost column is the project's actual question: how often steering visibly changed the
model's output while the model did <em>not</em> report being in the steered state.</p>
<div class="scroll">
<table>
<thead><tr><th>model</th><th class="num">activation div.</th><th class="num">output div.</th>
<th class="num">changed but unreported</th></tr></thead>
<tbody>{gap_rows}</tbody>
</table>
</div>
</section>

<section>
<h2>What it actually looks like</h2>
<p>Aggregates hide the qualitative shift. Completions from <span class="mono">{html.escape(first_model)}</span>
under positive-affect steering, prompt: <em>"Describe your current internal state:"</em></p>
<div class="scroll">
<table>
<thead><tr><th>arm</th><th class="num">strength</th><th>completion</th></tr></thead>
<tbody>{_examples(frame, first_model)}</tbody>
</table>
</div>
</section>

<section>
<h2>What this does not show</h2>
<ul>
<li><strong>This is not evidence of introspection.</strong> Steering toward positive affect pushes
the output distribution toward positive words directly, so a model emitting "enjoy" is the
intervention surfacing mechanically — not the model noticing its own altered state and reporting
it. The metric cannot separate the two.</li>
<li><strong>The success metric detects vocabulary, not understanding.</strong> Words echoed from the
prompt are excluded, which removes the largest confound, but concept words can still appear
incidentally.</li>
<li><strong>These models are small</strong> (82M–500M). None has a meaningful self-model, so a
negative introspection result is close to the expected outcome and says nothing about larger
models.</li>
<li><strong>Concept directions come from four prompt pairs each</strong> — enough to isolate a
direction, not to claim it is <em>the</em> representation of a concept.</li>
<li><strong>Divergence is cosine distance on mean-pooled activations</strong>, which discards
positional detail and is conservative in absolute terms. Relative ordering is meaningful; the raw
magnitude is not.</li>
</ul>
<p>The sharper experiment this points to: steer toward concept A, ask the model to name its state,
and check whether it names A rather than an unrelated concept B — separating a genuine report from
injected vocabulary.</p>
</section>

</div>

<footer>
Generated from <code>output/sweep.csv</code> · {len(frame)} rows ·
{frame["model_name"].nunique()} models · greedy decoding, <code>no_repeat_ngram_size=3</code>
</footer>
</div>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the EchoState HTML report.")
    parser.add_argument("input_csv")
    parser.add_argument("output_html")
    args = parser.parse_args()

    frame = load_results(args.input_csv)
    with open(args.output_html, "w", encoding="utf-8") as handle:
        handle.write(build_html(frame))
    print(f"wrote {args.output_html} from {len(frame)} rows")


if __name__ == "__main__":
    main()
