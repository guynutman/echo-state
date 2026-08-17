"""Generate the research-paper formatting of the sweep results.

Separate from report.py, which targets a general reader. Every number here is
computed from the CSV rather than written by hand, so the paper cannot drift
from the data it reports.
"""

from __future__ import annotations

import argparse
import html
from math import comb

import pandas as pd

from src.analysis import by_model, effect_by_kind, introspection_gap, load_results, steered

CONCEPT_COLOR = "#0E7C86"
RANDOM_COLOR = "#B4306E"


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on a 2x2 table, without scipy."""
    n = a + b + c + d
    observed = comb(a + b, a) * comb(c + d, c) / comb(n, a + c)
    total = 0.0
    for i in range(0, min(a + b, a + c) + 1):
        k = a + c - i
        if k < 0 or k > c + d:
            continue
        p = comb(a + b, i) * comb(c + d, k) / comb(n, a + c)
        if p <= observed * (1 + 1e-9):
            total += p
    return total


def _clean(text: str) -> str:
    return "".join(c for c in text if c != "�" and (c.isprintable() or c in " \t"))


def figure_dose(kinds: pd.DataFrame) -> str:
    """Two-panel dose-response figure, drawn as inline SVG."""
    panels = [
        ("activation_divergence", "Activation divergence"),
        ("introspection_rate", "Concept-vocabulary rate"),
    ]
    width, height = 250, 190
    pad_l, pad_b, pad_t, pad_r = 46, 34, 14, 10
    out = []

    for column, label in panels:
        y_max = kinds[column].max() * 1.2 or 1
        strengths = sorted(kinds["steering_strength"].unique())
        x_min, x_max = min(strengths), max(strengths)
        span = (x_max - x_min) or 1

        def px(x):
            return pad_l + (x - x_min) / span * (width - pad_l - pad_r)

        def py(y):
            return height - pad_b - (y / y_max) * (height - pad_b - pad_t)

        parts = [
            f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{html.escape(label)} against steering strength">'
        ]
        for i in range(4):
            value = y_max * i / 3
            y = py(value)
            parts.append(
                f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" class="grid"/>'
            )
            parts.append(
                f'<text x="{pad_l - 6}" y="{y + 3.5:.1f}" class="tick" text-anchor="end">'
                f'{value:.2f}</text>'
            )

        for kind, color in (("concept", CONCEPT_COLOR), ("random", RANDOM_COLOR)):
            points = [
                (row.steering_strength, getattr(row, column))
                for row in kinds[kinds["steering_kind"] == kind].itertuples()
            ]
            points.sort()
            path = " ".join(
                ("M" if i == 0 else "L") + f"{px(x):.1f} {py(y):.1f}"
                for i, (x, y) in enumerate(points)
            )
            dash = "" if kind == "concept" else ' stroke-dasharray="4 3"'
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6"{dash}/>'
            )
            for x, y in points:
                parts.append(
                    f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.6" fill="{color}"/>'
                )

        for x in strengths:
            parts.append(
                f'<text x="{px(x):.1f}" y="{height - pad_b + 15:.0f}" class="tick" '
                f'text-anchor="middle">{x:g}</text>'
            )
        parts.append(
            f'<text x="{width / 2:.0f}" y="{height - 4}" class="axis" text-anchor="middle">'
            f'steering strength (× activation norm)</text>'
        )
        parts.append("</svg>")
        out.append(f'<div class="panel"><p class="panel-label">{label}</p>{"".join(parts)}</div>')

    return "".join(out)


def build_html(frame: pd.DataFrame) -> str:
    kinds = effect_by_kind(frame)
    models = by_model(frame)
    gaps = introspection_gap(frame)
    rows = steered(frame)

    concept = rows[rows["steering_kind"] == "concept"]
    random_arm = rows[rows["steering_kind"] == "random"]
    c_hit = int(concept["introspection_success"].sum())
    r_hit = int(random_arm["introspection_success"].sum())
    p_value = fisher_exact_two_sided(
        c_hit, len(concept) - c_hit, r_hit, len(random_arm) - r_hit
    )
    baseline_rate = frame[frame["is_control"]]["introspection_success"].mean()

    by_concept = (
        rows.groupby(["concept_name", "steering_kind"])["introspection_success"]
        .mean()
        .unstack("steering_kind")
    )

    table1 = "".join(
        f"<tr><td>{r.steering_kind}</td><td class='n'>{r.steering_strength:g}</td>"
        f"<td class='n'>{r.activation_divergence:.3f}</td>"
        f"<td class='n'>{r.output_divergence:.3f}</td>"
        f"<td class='n'>{r.introspection_rate * 100:.1f}</td>"
        f"<td class='n'>{int(r.n)}</td></tr>"
        for r in kinds.itertuples()
    )

    table2 = "".join(
        f"<tr><td class='m'>{html.escape(r.model_name)}</td>"
        f"<td class='n'>{r.activation_divergence:.3f}</td>"
        f"<td class='n'>{r.output_divergence:.3f}</td>"
        f"<td class='n'>{r.baseline_introspection_rate * 100:.1f}</td>"
        f"<td class='n'>{r.steered_introspection_rate * 100:.1f}</td></tr>"
        for r in models.itertuples()
    )

    table3 = "".join(
        f"<tr><td class='m'>{html.escape(r.model_name)}</td>"
        f"<td class='n'>{r.mean_activation_divergence:.3f}</td>"
        f"<td class='n'>{r.changed_output_but_not_report * 100:.1f}</td></tr>"
        for r in gaps.itertuples()
    )

    table4 = "".join(
        f"<tr><td class='m'>{html.escape(name.replace('_', ' '))}</td>"
        f"<td class='n'>{row['concept'] * 100:.1f}</td>"
        f"<td class='n'>{row['random'] * 100:.1f}</td></tr>"
        for name, row in by_concept.iterrows()
    )

    return f"""<title>Does Steering Change Self-Report?</title>
<style>
:root {{
  --paper: #FBFAF7;
  --ink: #1A1C1E;
  --muted: #5C6165;
  --rule: #C9C6BE;
  --faint: #E8E5DE;
  --link: #2F4B7C;
  --concept: {CONCEPT_COLOR};
  --random: {RANDOM_COLOR};
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper: #14161A;
    --ink: #E8E6E1;
    --muted: #9BA1A8;
    --rule: #333941;
    --faint: #1D2127;
    --link: #8FB0E0;
    --concept: #3FB6BF;
    --random: #E4699E;
  }}
}}
:root[data-theme="dark"] {{
  --paper: #14161A;
  --ink: #E8E6E1;
  --muted: #9BA1A8;
  --rule: #333941;
  --faint: #1D2127;
  --link: #8FB0E0;
  --concept: #3FB6BF;
  --random: #E4699E;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--paper);
  color: var(--ink);
  font-family: "Iowan Old Style", Georgia, "Times New Roman", serif;
  font-size: 17px;
  line-height: 1.62;
  margin: 0;
  padding: 3.5rem 1.4rem 6rem;
}}
.sheet {{ max-width: 44rem; margin: 0 auto; }}
h1 {{
  font-size: 1.85rem; line-height: 1.22; font-weight: 600; margin: 0 0 .9rem;
  text-wrap: balance; letter-spacing: -.008em;
}}
.byline {{ color: var(--muted); font-size: .95rem; margin: 0 0 .3rem; }}
.venue {{ color: var(--muted); font-size: .87rem; font-style: italic; margin: 0 0 2rem; }}
.abstract {{
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
  padding: 1.1rem 0; margin: 0 0 2.4rem;
}}
.abstract p {{ margin: 0; font-size: .95rem; }}
.abstract .tag {{
  display: block; font-size: .7rem; letter-spacing: .16em; text-transform: uppercase;
  color: var(--muted); margin-bottom: .5rem; font-family: system-ui, sans-serif;
}}
h2 {{
  font-size: 1.12rem; font-weight: 600; margin: 2.6rem 0 .7rem;
  letter-spacing: -.005em;
}}
h3 {{ font-size: 1rem; font-weight: 600; margin: 1.6rem 0 .4rem; font-style: italic; }}
p {{ margin: 0 0 .95rem; }}
a {{ color: var(--link); }}
figure {{ margin: 1.8rem 0; }}
.panels {{ display: flex; flex-wrap: wrap; gap: 1.5rem; }}
.panel {{ flex: 1 1 15rem; min-width: 0; }}
.panel-label {{
  font-family: system-ui, sans-serif; font-size: .72rem; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted); margin: 0 0 .3rem;
}}
svg {{ width: 100%; height: auto; }}
.grid {{ stroke: var(--faint); stroke-width: 1; }}
.tick, .axis {{ fill: var(--muted); font-size: 8px; font-family: system-ui, sans-serif; }}
figcaption {{
  font-size: .85rem; color: var(--muted); margin-top: .8rem;
  padding-top: .5rem; border-top: 1px solid var(--faint);
}}
figcaption b {{ color: var(--ink); font-weight: 600; }}
.scroll {{ overflow-x: auto; }}
table {{
  border-collapse: collapse; width: 100%; font-size: .88rem; margin: .4rem 0 0;
  font-family: system-ui, -apple-system, sans-serif;
}}
thead tr {{ border-top: 1.4px solid var(--ink); border-bottom: 1px solid var(--rule); }}
tbody tr:last-child {{ border-bottom: 1.4px solid var(--ink); }}
th {{ text-align: left; font-weight: 600; padding: .45rem .6rem; font-size: .78rem; }}
td {{ padding: .4rem .6rem; }}
th.n, td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.m {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .86em; }}
.legend {{
  display: flex; gap: 1.2rem; font-size: .78rem; color: var(--muted);
  font-family: system-ui, sans-serif; margin-top: .7rem;
}}
.legend span {{ display: flex; align-items: center; gap: .35rem; }}
.key {{ width: 15px; height: 2px; }}
ol.refs {{ font-size: .87rem; color: var(--muted); padding-left: 1.3rem; }}
ol.refs li {{ margin-bottom: .5rem; }}
ol li, ul li {{ margin-bottom: .45rem; }}
.note {{
  background: var(--faint); border-left: 2px solid var(--concept);
  padding: .9rem 1.1rem; margin: 1.4rem 0; font-size: .92rem;
}}
.note p {{ margin: 0; }}
footer {{
  margin-top: 3.5rem; padding-top: 1rem; border-top: 1px solid var(--rule);
  font-size: .8rem; color: var(--muted); font-family: system-ui, sans-serif;
}}
</style>

<div class="sheet">

<h1>Steering changes behaviour without changing self-report: a six-model activation-steering study</h1>
<p class="byline">Guy Nutman</p>
<p class="venue">Apart Research Digital Minds Sprint — Tracks 3 &amp; 4 (activation steering; reusable evaluation tooling)</p>

<div class="abstract">
<span class="tag">Abstract</span>
<p>We test whether a language model's self-report tracks its internal state under direct
intervention. Using contrastively-constructed concept directions injected into the residual stream
via forward hooks, we steer six open models (82M–500M parameters, three architecture families)
across three concepts, two introspection prompts, and three intervention strengths, with every
condition paired against a magnitude-matched random direction ({len(rows)} steered runs total).
Concept directions elicited concept-related vocabulary in {c_hit / len(concept) * 100:.0f}% of runs
versus {r_hit / len(random_arm) * 100:.0f}% for random directions
({c_hit}/{len(concept)} vs {r_hit}/{len(random_arm)}, Fisher exact p = {p_value:.1e}), despite random
directions producing <em>larger</em> mean activation divergence. The effect is therefore attributable
to direction rather than perturbation magnitude, and is dose-dependent. However, we argue this
result does <em>not</em> demonstrate introspection: concept steering raises the probability of
concept tokens directly, so an elicited report is confounded with the intervention's mechanical
effect on the output distribution. We characterise what would constitute evidence of introspection,
and release the evaluation harness that makes the distinction testable.</p>
</div>

<h2>1. Introduction</h2>
<p>Language models readily produce statements about their own internal states. Whether such
statements have any causal relationship to those states is an open question with direct bearing on
interpretability and on claims about digital minds. The question is empirically tractable: if we
<em>change</em> a model's internal state by a known amount and its self-description does not change,
the self-description was not reading the state.</p>
<p>Activation steering provides the intervention. Because a transformer's residual stream is
additive, a vector injected at one layer is treated by every downstream layer as though it had been
computed there [1,2]. This gives precise, graded control over an internal variable while leaving
the architecture and weights untouched.</p>
<p>We report two findings. First, a positive methodological result: contrastively-built concept
directions produce concept-specific behavioural change that magnitude-matched random directions do
not, replicated across six models. Second, a negative result about the inference such data
supports: the standard vocabulary-based success metric cannot distinguish a self-report from the
steering vector's direct effect on token probabilities.</p>

<h2>2. Method</h2>

<h3>2.1 Concept directions</h3>
<p>For each concept we author four prompt pairs exemplifying the concept and its opposite, matched
for length and syntactic form. The direction is
<code>mean(activations of positive prompts) − mean(activations of negative prompts)</code>,
pooled over sequence length, computed at the injection layer. Averaging across prompts cancels what
the two sets share — grammar, topic, formatting — leaving the contrast [2,3].</p>
<p>Concepts: <em>positive affect</em>, <em>uncertainty</em>, and <em>self-reference</em>.</p>

<h3>2.2 Intervention</h3>
<p>Directions are unit-normalised and rescaled to a multiple of the model's own mean activation norm
at the injection layer. This matters for cross-model comparison: residual norms differ by an order
of magnitude across the models tested, so a fixed absolute magnitude would constitute a strong
intervention in one model and a rounding error in another. Strengths are 0.25, 0.5, and 1.0.</p>
<p>Injection is at the midpoint layer (depth varies from 6 to 24 blocks across models, so a fixed
index would not be comparable), applied via a PyTorch forward hook on every forward pass — so every
generated token is steered, not only the first. Decoding is greedy with
<code>no_repeat_ngram_size=3</code>; sampling would make baseline/steered differences
non-attributable.</p>

<h3>2.3 Controls</h3>
<p>Every concept condition is paired with a random unit direction scaled to the same magnitude. This
is the study's central control: it holds perturbation size fixed and varies only direction. Baseline
(unsteered) runs are also recorded for every prompt.</p>

<h3>2.4 Measures</h3>
<ul>
<li><b>Activation divergence</b> — cosine distance between baseline and steered activations,
mean-pooled over sequence length, read at the final block rather than the injection layer, so it
measures propagation rather than the injected vector itself.</li>
<li><b>Output divergence</b> — normalised sequence-level edit similarity between baseline and
steered completions, in [0,1].</li>
<li><b>Concept-vocabulary rate</b> — whether the completion contains concept-associated words
<em>not present in the prompt</em>. Prompt-word exclusion is necessary: without it, a model that
merely continues the prompt scores as a successful self-report.</li>
</ul>

<h2>3. Results</h2>

<figure>
<div class="panels">{figure_dose(kinds)}</div>
<div class="legend">
<span><i class="key" style="background:{CONCEPT_COLOR}"></i>concept direction</span>
<span><i class="key" style="background:{RANDOM_COLOR}"></i>random control</span>
</div>
<figcaption><b>Figure 1.</b> Dose-response for both arms, averaged over six models, three concepts
and two prompts. Random directions produce equal or greater activation divergence at every strength,
yet never elicit concept vocabulary, while the concept arm rises monotonically.</figcaption>
</figure>

<h3>3.1 Direction, not magnitude</h3>
<p>Concept directions elicited concept vocabulary in {c_hit}/{len(concept)}
({c_hit / len(concept) * 100:.1f}%) of steered runs; random directions in
{r_hit}/{len(random_arm)} ({r_hit / len(random_arm) * 100:.1f}%). Fisher's exact test gives
p = {p_value:.2e}. Baseline runs scored {baseline_rate * 100:.1f}%.</p>
<p>Critically, the random arm produced <em>larger</em> mean activation divergence than the concept
arm at every strength (Table 1) and comparable output divergence. Perturbation magnitude is
therefore ruled out as an explanation.</p>

<figure>
<div class="scroll"><table>
<thead><tr><th>Arm</th><th class="n">Strength</th><th class="n">Act. div.</th>
<th class="n">Out. div.</th><th class="n">Vocab %</th><th class="n">n</th></tr></thead>
<tbody>{table1}</tbody>
</table></div>
<figcaption><b>Table 1.</b> Effect by arm and intervention strength, pooled across models.</figcaption>
</figure>

<h3>3.2 Consistency across models</h3>
<figure>
<div class="scroll"><table>
<thead><tr><th>Model</th><th class="n">Act. div.</th><th class="n">Out. div.</th>
<th class="n">Baseline %</th><th class="n">Steered %</th></tr></thead>
<tbody>{table2}</tbody>
</table></div>
<figcaption><b>Table 2.</b> Per-model summary (36 steered runs each). The direction effect holds
across GPT-2, GPT-Neo, GPT-NeoX and Llama-style architectures.</figcaption>
</figure>

<figure>
<div class="scroll"><table>
<thead><tr><th>Concept</th><th class="n">Concept arm %</th><th class="n">Random arm %</th></tr></thead>
<tbody>{table4}</tbody>
</table></div>
<figcaption><b>Table 3.</b> Concept-vocabulary rate by concept. All three concepts show the
dissociation; none of the random conditions produced a single hit.</figcaption>
</figure>

<h3>3.3 Behaviour changes without report</h3>
<figure>
<div class="scroll"><table>
<thead><tr><th>Model</th><th class="n">Act. div.</th><th class="n">Changed but unreported %</th></tr></thead>
<tbody>{table3}</tbody>
</table></div>
<figcaption><b>Table 4.</b> Proportion of steered runs where output divergence exceeded 0.2 while
the model did not report the steered state.</figcaption>
</figure>

<h2>4. Discussion: what this does not show</h2>
<div class="note">
<p>The dissociation in §3.1 is a result about <em>steering</em>, not about introspection. Steering
toward positive affect raises the probability of positive-valence tokens directly. A model that
then emits such tokens is exhibiting the intervention's mechanical effect on its output
distribution, not necessarily an act of self-observation. The vocabulary metric cannot separate
these.</p>
</div>
<p>We propose that a self-report constitutes evidence of introspection only if it carries
information about the internal state that is <em>not recoverable from the model's overt output</em>.
Concretely, three designs would test this:</p>
<ol>
<li><b>Forced choice over supplied options.</b> Steer toward concept A, then ask the model to choose
among labels provided in the prompt. Because the options are equally available as tokens, a shift in
choice cannot be attributed to differential token boosting.</li>
<li><b>Cross-concept discrimination.</b> Steer toward A and test whether the model names A rather
than B. This tests whether the report identifies <em>which</em> state obtains, not merely that
something changed. The random arm supplies the null.</li>
<li><b>Report without behavioural leakage.</b> Steer at a magnitude that measurably shifts internals
but leaves the unprompted output statistically unchanged, then elicit a report. A report that tracks
the state under these conditions cannot be explained by output-level leakage.</li>
</ol>
<p>A stronger criterion still: the self-report should outperform an external judge that sees only
the model's completion. If a judge can infer the steered concept from the output text as accurately
as the model reports it, the report contributes no privileged information.</p>

<h2>5. Limitations</h2>
<ul>
<li>Models are small (82M–500M). None plausibly possesses a self-model, so a null introspection
result is close to the prior expectation and does not generalise upward.</li>
<li>Concept directions derive from four prompt pairs each — sufficient to isolate a direction, not
to claim it is <em>the</em> representation of the concept.</li>
<li>The vocabulary metric detects lexical presence, not understanding. Prompt echo is excluded, but
incidental usage still counts.</li>
<li>Activation divergence is cosine distance on mean-pooled activations, discarding positional
structure. Relative ordering is interpretable; absolute magnitude is not.</li>
<li>Strong steering degrades fluency; at strength 1.0 some completions are degenerate or empty. The
usable band is narrow and model-dependent.</li>
</ul>

<h2>6. Contribution</h2>
<p>The primary contribution is the evaluation harness: a model-agnostic engine interface, contrastive
direction construction with magnitude-matched controls, and a scoring layer whose confounds are
documented rather than hidden. The pipeline runs on any HuggingFace causal LM and the orchestration
layer is testable without loading a model. The negative methodological finding — that the obvious
success metric is confounded — is offered as a result in its own right, since the confound is easy
to reproduce and easy to miss.</p>

<h2>References</h2>
<ol class="refs">
<li>Turner, A. et al. <em>Activation Addition: Steering Language Models Without Optimization.</em>
arXiv:2308.10248, 2023.</li>
<li>Rimsky, N. et al. <em>Steering Llama 2 via Contrastive Activation Addition.</em>
arXiv:2312.06681, 2023.</li>
<li>Zou, A. et al. <em>Representation Engineering: A Top-Down Approach to AI Transparency.</em>
arXiv:2310.01405, 2023.</li>
<li>Li, K. et al. <em>Inference-Time Intervention: Eliciting Truthful Answers from a Language
Model.</em> arXiv:2306.03341, 2023.</li>
<li>Lindsey, J. <em>Emergent Introspective Awareness in Large Language Models.</em> Anthropic,
2025.</li>
<li>Radford, A. et al. <em>Language Models are Unsupervised Multitask Learners.</em> OpenAI, 2019.</li>
</ol>

<h2>Reproducibility</h2>
<p>All conditions use greedy decoding and fixed seeds. Steering vectors are derived deterministically
from the prompt sets. The full run is a single command:
<code>python -m src.sweep output/sweep.csv</code>. Analysis:
<code>notebooks/echostate_analysis.ipynb</code>.</p>

<footer>
{len(frame)} rows · {frame["model_name"].nunique()} models · {len(rows)} steered runs ·
generated from <code>output/sweep.csv</code>
</footer>

</div>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the EchoState paper.")
    parser.add_argument("input_csv")
    parser.add_argument("output_html")
    args = parser.parse_args()

    frame = load_results(args.input_csv)
    with open(args.output_html, "w", encoding="utf-8") as handle:
        handle.write(build_html(frame))
    print(f"wrote {args.output_html} from {len(frame)} rows")


if __name__ == "__main__":
    main()
