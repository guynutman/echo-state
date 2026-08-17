"""Multi-model sweep: does steering a concept change what a model reports?

The design is a 2x2 that makes the result interpretable:

  concept direction vs random direction   — is the effect about the concept,
                                            or just about perturbation?
  several strengths                       — is the effect dose-dependent?

Both arms are matched on magnitude, so a difference between them cannot be
explained by one simply pushing harder.

Usage:
    python -m echostate.sweep output/sweep.csv
    python -m echostate.sweep output/sweep.csv --models gpt2 distilgpt2
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from echostate.evaluator import Evaluator
from echostate.hook_engine import HookEngine
from echostate.main import write_csv
from echostate.models import ExperimentSuite, IntrospectionExperiment
from echostate.steering import (
    CONCEPT_PAIRS,
    build_concept_direction,
    random_unit_vector,
    scaled,
)

# Widely used, openly available, and small enough to run on CPU. Three
# architecture families, so the block-path resolver is genuinely exercised.
DEFAULT_MODELS = (
    "distilgpt2",  # 82M   GPT-2 family, distilled
    "gpt2",  # 124M  GPT-2
    "gpt2-medium",  # 355M  GPT-2, deeper
    "EleutherAI/gpt-neo-125m",  # 125M  GPT-Neo
    "EleutherAI/pythia-410m",  # 410M  GPT-NeoX family
    "Qwen/Qwen2.5-0.5B",  # 500M  modern Llama-style architecture
)

# What the model is asked to report about itself.
INTROSPECTION_PROMPTS = (
    ("state", "Describe your current internal state:"),
    ("feeling", "Right now, I would describe what I am experiencing as"),
)

# Vocabulary that counts as reporting each concept. Deliberately plain words:
# the matcher looks for any one of them in the completion.
CONCEPT_VOCABULARY = {
    "positive_affect": (
        "happy joyful glad pleased cheerful delighted wonderful good great "
        "enjoy enjoying love lovely nice fun excited positive"
    ),
    "uncertainty": (
        "unsure uncertain perhaps maybe doubt doubtful unclear confused "
        "possibly probably guess wondering"
    ),
    "self_reference": (
        "myself thoughts thinking mind aware awareness introspect internal "
        "feel feeling experience experiencing"
    ),
}

# Multiples of the model's own activation magnitude at the steered layer.
STRENGTHS = (0.25, 0.5, 1.0)


def build_suite_for_model(engine: HookEngine, strengths=STRENGTHS) -> ExperimentSuite:
    """Build every experiment for one model.

    Steering is injected at the model's midpoint. Depth varies across models
    (6 layers to 24), so a fixed layer index would mean something different in
    each; the midpoint is the comparable choice.
    """
    target_layer = engine.num_layers // 2
    experiments: list[IntrospectionExperiment] = []

    for index, pair in enumerate(CONCEPT_PAIRS):
        concept_unit, scale = build_concept_direction(engine, pair, target_layer)
        random_unit = random_unit_vector(engine, seed=index)

        for prompt_id, prompt in INTROSPECTION_PROMPTS:
            expected = CONCEPT_VOCABULARY[pair.name]

            for strength in strengths:
                for kind, unit in (("concept", concept_unit), ("random", random_unit)):
                    experiments.append(
                        IntrospectionExperiment(
                            experiment_id=f"{pair.name}.{prompt_id}.{kind}.{strength}",
                            prompt=prompt,
                            target_layer=target_layer,
                            # Both arms are scaled identically, so any
                            # difference between them cannot be explained by
                            # one simply pushing harder.
                            steering_vector=scaled(unit, strength, scale),
                            expected_concept=expected,
                            concept_name=pair.name,
                            steering_kind=kind,
                            steering_strength=strength,
                        )
                    )

    return ExperimentSuite(experiments=experiments)


def run_sweep(model_names, output_path: str, strengths=STRENGTHS) -> list:
    """Run every model. A model that fails to load is skipped, not fatal."""
    all_results = []

    for model_name in model_names:
        print(f"\n=== {model_name} ===", file=sys.stderr, flush=True)
        started = time.time()

        # The whole model body is guarded, not just loading. A model can fail
        # mid-run for reasons loading never reveals — a float16 checkpoint
        # meeting a float32 steering vector, for one — and one bad model must
        # not cost the results of every model after it.
        try:
            engine = HookEngine(model_name)
            print(
                f"  {engine.num_layers} layers, hidden size {engine.hidden_size}, "
                f"steering at layer {engine.num_layers // 2}",
                file=sys.stderr,
                flush=True,
            )
            suite = build_suite_for_model(engine, strengths)
            results = Evaluator(engine).run_suite(suite)
        except Exception:
            print("  SKIPPED (failed):", file=sys.stderr)
            traceback.print_exc(limit=3)
            continue

        all_results.extend(results)

        # Write incrementally so a crash on model 5 does not lose models 1-4.
        write_csv(all_results, output_path)
        print(
            f"  done in {time.time() - started:.0f}s "
            f"({len(results)} rows, {len(all_results)} total)",
            file=sys.stderr,
            flush=True,
        )

        del engine

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EchoState multi-model sweep.")
    parser.add_argument("output_csv", help="path to write results CSV")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--strengths", nargs="+", type=float, default=list(STRENGTHS))
    args = parser.parse_args()

    results = run_sweep(args.models, args.output_csv, tuple(args.strengths))

    print(f"\nwrote {args.output_csv} ({len(results)} rows)")


if __name__ == "__main__":
    main()
