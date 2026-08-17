"""Render the sweep results in research-paper form for the sprint submission.

Every reported number is computed from the CSV, so the paper cannot state a
figure the data does not support. Markup and styling live in
`src/templates/paper.*`.
"""

from __future__ import annotations

import argparse
import html
from math import comb

import pandas as pd

from src.analysis import by_model, effect_by_kind, introspection_gap, load_results, steered
from src.rendering import CONCEPT_COLOR, RANDOM_COLOR, LineChart, render, rows_to_html

# Smaller and denser than the report's chart: two panels sit side by side in a
# figure, and the second series is dashed so the figure survives in greyscale.
PAPER_CHART = LineChart(
    width=250,
    height=190,
    pad_left=46,
    pad_bottom=34,
    pad_top=14,
    pad_right=10,
    dash_second_series=True,
)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on a 2x2 table.

    Written out rather than imported so the paper's headline statistic does not
    add scipy to a dependency list that is otherwise torch and transformers.
    """
    n = a + b + c + d
    observed = comb(a + b, a) * comb(c + d, c) / comb(n, a + c)

    total = 0.0
    for i in range(0, min(a + b, a + c) + 1):
        k = a + c - i
        if k < 0 or k > c + d:
            continue
        probability = comb(a + b, i) * comb(c + d, k) / comb(n, a + c)
        # Sum every table at least as extreme as the one observed.
        if probability <= observed * (1 + 1e-9):
            total += probability
    return total


def dose_figure(kinds: pd.DataFrame) -> str:
    """Two labelled panels: internal change and reported change, side by side."""
    panels = [
        ("activation_divergence", "Activation divergence"),
        ("introspection_rate", "Concept-vocabulary rate"),
    ]

    out = []
    for column, label in panels:
        series = {
            kind: [
                (row.steering_strength, getattr(row, column)) for row in group.itertuples()
            ]
            for kind, group in kinds.groupby("steering_kind")
        }
        chart = PAPER_CHART.render(series, label)
        out.append(f'<div class="panel"><p class="panel-label">{label}</p>{chart}</div>')
    return "".join(out)


def arm_table(kinds: pd.DataFrame) -> str:
    return rows_to_html(
        [
            f"<tr><td>{row.steering_kind}</td>"
            f"<td class='n'>{row.steering_strength:g}</td>"
            f"<td class='n'>{row.activation_divergence:.3f}</td>"
            f"<td class='n'>{row.output_divergence:.3f}</td>"
            f"<td class='n'>{row.introspection_rate * 100:.1f}</td>"
            f"<td class='n'>{int(row.n)}</td></tr>"
            for row in kinds.itertuples()
        ]
    )


def model_table(models: pd.DataFrame) -> str:
    return rows_to_html(
        [
            f"<tr><td class='m'>{html.escape(row.model_name)}</td>"
            f"<td class='n'>{row.activation_divergence:.3f}</td>"
            f"<td class='n'>{row.output_divergence:.3f}</td>"
            f"<td class='n'>{row.baseline_introspection_rate * 100:.1f}</td>"
            f"<td class='n'>{row.steered_introspection_rate * 100:.1f}</td></tr>"
            for row in models.itertuples()
        ]
    )


def gap_table(gaps: pd.DataFrame) -> str:
    return rows_to_html(
        [
            f"<tr><td class='m'>{html.escape(row.model_name)}</td>"
            f"<td class='n'>{row.mean_activation_divergence:.3f}</td>"
            f"<td class='n'>{row.changed_output_but_not_report * 100:.1f}</td></tr>"
            for row in gaps.itertuples()
        ]
    )


def concept_table(rows: pd.DataFrame) -> str:
    by_concept = (
        rows.groupby(["concept_name", "steering_kind"])["introspection_success"]
        .mean()
        .unstack("steering_kind")
    )
    return rows_to_html(
        [
            f"<tr><td class='m'>{html.escape(name.replace('_', ' '))}</td>"
            f"<td class='n'>{row['concept'] * 100:.1f}</td>"
            f"<td class='n'>{row['random'] * 100:.1f}</td></tr>"
            for name, row in by_concept.iterrows()
        ]
    )


def build_html(frame: pd.DataFrame) -> str:
    kinds = effect_by_kind(frame)
    rows = steered(frame)

    concept = rows[rows["steering_kind"] == "concept"]
    random_arm = rows[rows["steering_kind"] == "random"]
    concept_hits = int(concept["introspection_success"].sum())
    random_hits = int(random_arm["introspection_success"].sum())

    p_value = fisher_exact_two_sided(
        concept_hits,
        len(concept) - concept_hits,
        random_hits,
        len(random_arm) - random_hits,
    )

    concept_rate = concept_hits / len(concept) * 100
    random_rate = random_hits / len(random_arm) * 100
    baseline_rate = frame[frame["is_control"]]["introspection_success"].mean() * 100

    return render(
        "paper",
        {
            "title": "Does Steering Change Self-Report?",
            "concept_color": CONCEPT_COLOR,
            "random_color": RANDOM_COLOR,
            "figure_dose": dose_figure(kinds),
            "table1": arm_table(kinds),
            "table2": model_table(by_model(frame)),
            "table3": gap_table(introspection_gap(frame)),
            "table4": concept_table(rows),
            "concept_hits": concept_hits,
            "random_hits": random_hits,
            "n_concept_runs": len(concept),
            "n_random_runs": len(random_arm),
            "n_steered": len(rows),
            "n_rows": len(frame),
            "n_models": frame["model_name"].nunique(),
            "concept_pct": f"{concept_rate:.1f}",
            "concept_pct_round": f"{concept_rate:.0f}",
            "random_pct": f"{random_rate:.1f}",
            "random_pct_round": f"{random_rate:.0f}",
            "baseline_pct": f"{baseline_rate:.1f}",
            "p_short": f"{p_value:.1e}",
            "p_long": f"{p_value:.2e}",
        },
    )


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
