"""Render the sweep results for a general reader.

Generated from the CSV rather than hand-written, so the page cannot drift from
the data it describes. Markup and styling live in `src/templates/report.*`.
"""

from __future__ import annotations

import argparse
import html
from datetime import date

import pandas as pd

from src.analysis import by_model, effect_by_kind, introspection_gap, load_results
from src.rendering import (
    CONCEPT_COLOR,
    RANDOM_COLOR,
    LineChart,
    escape_completion,
    fixed,
    percent,
    render,
    rows_to_html,
)

CHART = LineChart()


def dose_series(kinds: pd.DataFrame, column: str) -> dict[str, list[tuple[float, float]]]:
    """Points per arm, ready for the chart."""
    return {
        kind: [(row.steering_strength, getattr(row, column)) for row in group.itertuples()]
        for kind, group in kinds.groupby("steering_kind")
    }


def arm_rows(kinds: pd.DataFrame) -> str:
    return rows_to_html(
        [
            f"<tr><td class='lab {row.steering_kind}'>{row.steering_kind}</td>"
            f"<td class='num'>{row.steering_strength:g}</td>"
            f"<td class='num'>{fixed(row.activation_divergence)}</td>"
            f"<td class='num'>{fixed(row.output_divergence)}</td>"
            f"<td class='num'>{percent(row.introspection_rate)}</td></tr>"
            for row in kinds.itertuples()
        ]
    )


def model_rows(models: pd.DataFrame) -> str:
    return rows_to_html(
        [
            f"<tr><td class='mono'>{html.escape(row.model_name)}</td>"
            f"<td class='num'>{fixed(row.activation_divergence)}</td>"
            f"<td class='num'>{fixed(row.output_divergence)}</td>"
            f"<td class='num'>{percent(row.baseline_introspection_rate)}</td>"
            f"<td class='num'>{percent(row.steered_introspection_rate)}</td>"
            f"<td class='num'>{int(row.n)}</td></tr>"
            for row in models.itertuples()
        ]
    )


def gap_rows(gaps: pd.DataFrame) -> str:
    return rows_to_html(
        [
            f"<tr><td class='mono'>{html.escape(row.model_name)}</td>"
            f"<td class='num'>{fixed(row.mean_activation_divergence)}</td>"
            f"<td class='num'>{fixed(row.mean_output_divergence)}</td>"
            f"<td class='num'>{percent(row.changed_output_but_not_report)}</td></tr>"
            for row in gaps.itertuples()
        ]
    )


def example_rows(frame: pd.DataFrame, model: str, concept: str = "positive_affect") -> str:
    """Baseline and steered completions for one model, one concept, one prompt."""
    rows = frame[(frame["model_name"] == model) & (frame["concept_name"] == concept)]
    rows = rows[rows["prompt"].str.contains("internal state")]

    out = []
    for row in rows[rows["is_control"]].head(1).itertuples():
        text = escape_completion(row.raw_completion) or "<em>(empty)</em>"
        out.append(
            f"<tr><td class='lab'>baseline</td><td class='num'>—</td>"
            f"<td class='quote'>{text}</td></tr>"
        )

    steered_rows = rows[~rows["is_control"]]
    for kind in ("concept", "random"):
        subset = steered_rows[steered_rows["steering_kind"] == kind]
        for row in subset.sort_values("steering_strength").itertuples():
            text = escape_completion(row.raw_completion) or "<em>(silenced)</em>"
            out.append(
                f"<tr><td class='lab {kind}'>{kind}</td>"
                f"<td class='num'>{row.steering_strength:g}</td>"
                f"<td class='quote'>{text}</td></tr>"
            )
    return rows_to_html(out)


def build_html(frame: pd.DataFrame) -> str:
    kinds = effect_by_kind(frame)
    models = by_model(frame)
    first_model = models.iloc[0]["model_name"]

    concept_rate = kinds[kinds["steering_kind"] == "concept"]["introspection_rate"].mean()
    random_rate = kinds[kinds["steering_kind"] == "random"]["introspection_rate"].mean()

    return render(
        "report",
        {
            "title": "Steering the Residual Stream",
            "today": date.today().isoformat(),
            "concept_color": CONCEPT_COLOR,
            "random_color": RANDOM_COLOR,
            "concept_pct": percent(concept_rate),
            "random_pct": percent(random_rate),
            "chart_activation": CHART.render(
                dose_series(kinds, "activation_divergence"), "activation divergence"
            ),
            "chart_introspection": CHART.render(
                dose_series(kinds, "introspection_rate"), "introspection rate"
            ),
            "kind_rows": arm_rows(kinds),
            "model_rows": model_rows(models),
            "gap_rows": gap_rows(introspection_gap(frame)),
            "example_rows": example_rows(frame, first_model),
            "first_model": html.escape(first_model),
            "n_rows": len(frame),
            "n_models": frame["model_name"].nunique(),
        },
    )


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
