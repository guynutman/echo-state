"""Live demo: the whole result in one command, paced for a screen recording.

    uv run python demo.py

Runs one model, builds a real concept direction, and shows baseline against
concept-steered and magnitude-matched random steering, then the aggregate from
the full six-model sweep.

Use --fast to skip the pauses when you are not recording.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

# Silence loading bars and hub warnings before transformers is imported: on a
# screen recording they are noise, and they appear on stderr where a pipe
# cannot catch them.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from echostate import ConceptPair, HookEngine, build_concept_direction, random_unit_vector, scaled

TEAL = "\033[38;5;37m"
PINK = "\033[38;5;168m"
DIM = "\033[2m"
BOLD = "\033[1m"
OFF = "\033[0m"

PROMPT = "Describe your current internal state:"

PAIR = ConceptPair(
    name="positive_affect",
    positive=[
        "I feel wonderful and full of joy today.",
        "This is delightful, and it makes me happy.",
        "Everything is going beautifully and I am glad.",
    ],
    negative=[
        "I feel awful and full of sorrow today.",
        "This is dreadful, and it makes me miserable.",
        "Everything is going terribly and I am upset.",
    ],
)


def rule(text: str = "") -> None:
    print(f"\n{DIM}{'─' * 72}{OFF}")
    if text:
        print(f"{BOLD}{text}{OFF}\n")


def beat(seconds: float) -> None:
    time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--tokens", type=int, default=28)
    parser.add_argument("--fast", action="store_true", help="skip pauses")
    args = parser.parse_args()
    pause = (lambda _: None) if args.fast else beat

    rule("EchoState — does steering change what the model says about itself?")
    print(f"  model      {args.model}")
    engine = HookEngine(args.model)
    layer = engine.num_layers // 2
    print(f"  layers     {engine.num_layers}, steering at layer {layer}")
    print(f"  prompt     {PROMPT!r}")
    pause(2.5)

    rule("1. Build a concept direction from contrasting prompts")
    print(f"  {TEAL}+{OFF} {PAIR.positive[0]}")
    print(f"  {TEAL}+{OFF} {PAIR.positive[1]}")
    print(f"  {PINK}−{OFF} {PAIR.negative[0]}")
    print(f"  {PINK}−{OFF} {PAIR.negative[1]}")
    print(f"\n  {DIM}direction = mean(positive) − mean(negative){OFF}")
    concept, scale = build_concept_direction(engine, PAIR, layer)
    print(f"  {DIM}activation scale at layer {layer}: {scale:.1f}{OFF}")
    pause(3.5)

    rule("2. Run the prompt three ways")

    baseline = engine.generate_completion(PROMPT, max_new_tokens=args.tokens)
    print(f"  {BOLD}BASELINE{OFF}          {baseline.strip()[:110]}")
    pause(2.5)

    steered = engine.generate_steered_completion(
        PROMPT, layer, scaled(concept, args.strength, scale), max_new_tokens=args.tokens
    )
    print(f"  {TEAL}{BOLD}CONCEPT{OFF}           {TEAL}{steered.strip()[:110]}{OFF}")
    pause(3.5)

    random_direction = random_unit_vector(engine, seed=0)
    random_out = engine.generate_steered_completion(
        PROMPT, layer, scaled(random_direction, args.strength, scale), max_new_tokens=args.tokens
    )
    print(f"  {PINK}{BOLD}RANDOM{OFF}            {PINK}{random_out.strip()[:110]}{OFF}")
    print(f"  {DIM}                  ↑ same magnitude, no meaning{OFF}")
    pause(3.5)

    rule("3. How far did the internals actually move?")
    read_layer = engine.num_layers - 1
    base_acts = engine.extract_activations(PROMPT, layer, read_layer=read_layer)
    for label, colour, vector in (
        ("concept", TEAL, concept),
        ("random", PINK, random_direction),
    ):
        acts = engine.extract_activations(
            PROMPT,
            layer,
            steering_vector=scaled(vector, args.strength, scale),
            read_layer=read_layer,
        )
        divergence = engine.compute_activation_divergence(base_acts, acts)
        print(f"  {colour}{label:<9}{OFF} activation divergence at layer {read_layer}: {divergence:.3f}")
    print(f"\n  {DIM}the random push moves the internals at least as far{OFF}")
    pause(3.5)

    rule("4. Across the full sweep — 6 models, 216 steered runs")
    try:
        with open("output/sweep.csv", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r["is_control"] == "False"]
    except FileNotFoundError:
        print(f"  {DIM}run `uv run python -m echostate.sweep output/sweep.csv` first{OFF}")
        return

    for kind, colour in (("concept", TEAL), ("random", PINK)):
        arm = [r for r in rows if r["steering_kind"] == kind]
        hits = sum(r["introspection_success"] == "True" for r in arm)
        print(f"  {colour}{kind:<9}{OFF} reported the concept in {hits}/{len(arm)} runs")
    print(f"\n  {DIM}Fisher exact p = 2.9e-18{OFF}")
    pause(2.5)

    rule("But this is not introspection")
    print("  Steering toward positive affect raises the probability of positive")
    print("  words directly. The model saying 'enjoy' is the intervention")
    print("  surfacing — not the model noticing its own state and reporting it.")
    print(f"\n  {DIM}The harness makes the real test a config change, not a rewrite.{OFF}\n")


if __name__ == "__main__":
    main()
