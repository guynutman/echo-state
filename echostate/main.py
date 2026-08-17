"""CLI entry point: read an experiment JSON, run it, write a results CSV.

Usage:
    python -m echostate.main experiments/sample_experiments.json output/results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from echostate.evaluator import Evaluator
from echostate.hook_engine import HookEngine
from echostate.models import ArtifactResult, ExperimentSuite

CSV_COLUMNS = [
    "model_name",
    "experiment_id",
    "concept_name",
    "steering_kind",
    "steering_strength",
    "is_control",
    "introspection_success",
    "activation_divergence",
    "output_divergence",
    "target_layer",
    "read_layer",
    "prompt",
    "raw_completion",
]


def load_experiments(path: str) -> ExperimentSuite:
    """Load and validate an experiment suite. Bad input fails here, not in a hook."""
    return ExperimentSuite.model_validate_json(Path(path).read_text())


def write_csv(results: list[ArtifactResult], output_path: str) -> None:
    """Write results to CSV, creating the output directory if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # newline="" is required by the csv module: without it, completions
    # containing newlines produce stray blank lines on some platforms.
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.model_dump(include=set(CSV_COLUMNS)))


def summarize(results: list[ArtifactResult]) -> str:
    """Human-readable summary for stdout."""
    controls = [r for r in results if r.is_control]
    steered = [r for r in results if not r.is_control]

    def success_rate(rows: list[ArtifactResult]) -> str:
        if not rows:
            return "n/a"
        hits = sum(r.introspection_success for r in rows)
        return f"{hits}/{len(rows)} ({hits / len(rows):.0%})"

    lines = [
        f"rows written:            {len(results)}",
        f"control introspection:   {success_rate(controls)}",
        f"steered introspection:   {success_rate(steered)}",
    ]
    if steered:
        divergences = [r.activation_divergence for r in steered]
        lines.append(
            f"activation divergence:   min {min(divergences):.5f} / "
            f"max {max(divergences):.5f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EchoState introspection experiments.")
    parser.add_argument("input_json", help="path to an experiment suite JSON file")
    parser.add_argument("output_csv", help="path to write results CSV")
    parser.add_argument(
        "--model", default="gpt2", help="HuggingFace model name (default: gpt2)"
    )
    args = parser.parse_args()

    suite = load_experiments(args.input_json)
    print(f"loaded {len(suite.experiments)} experiments", file=sys.stderr)

    print(f"loading {args.model}...", file=sys.stderr)
    evaluator = Evaluator(HookEngine(args.model))

    results = evaluator.run_suite(suite)
    write_csv(results, args.output_csv)

    print(f"\nwrote {args.output_csv}")
    print(summarize(results))


if __name__ == "__main__":
    main()
