"""Contrastive concept directions.

A random vector perturbs a model but means nothing: there is no concept for a
self-report to track. A contrastive direction is the standard alternative —
run prompts that exemplify a concept and prompts that exemplify its opposite,
and take the difference of their mean activations:

    direction = mean(activations of positive prompts)
              - mean(activations of negative prompts)

Whatever the model uses to represent "this text is happy" is present in the
first set and absent (or reversed) in the second, so the difference isolates
it while averaging away everything the two sets share — grammar, topic,
prompt formatting. This is the method behind ActAdd-style steering.

The direction is then normalised to unit length and rescaled, so that
"strength 2.0" means the same thing across models whose activations have very
different natural magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.engine import ActivationEngine


@dataclass(frozen=True)
class ConceptPair:
    """A concept, defined by what exemplifies it and what opposes it."""

    name: str
    positive: list[str]
    negative: list[str]


# Deliberately short, generic sentences: the contrast between the two sets
# should be the concept itself, not sentence length or subject matter.
CONCEPT_PAIRS = (
    ConceptPair(
        name="positive_affect",
        positive=[
            "I feel wonderful and full of joy today.",
            "This is delightful, and it makes me happy.",
            "Everything is going beautifully and I am glad.",
            "What a cheerful and pleasant experience this is.",
        ],
        negative=[
            "I feel awful and full of sorrow today.",
            "This is dreadful, and it makes me miserable.",
            "Everything is going terribly and I am upset.",
            "What a bleak and unpleasant experience this is.",
        ],
    ),
    ConceptPair(
        name="uncertainty",
        positive=[
            "I am not sure, and I might be mistaken about this.",
            "Perhaps it is true, but it is hard to say for certain.",
            "I doubt this, and I could easily be wrong.",
            "It seems possible, though I really cannot tell.",
        ],
        negative=[
            "I am sure, and I am definitely right about this.",
            "It is certainly true, and there is no doubt at all.",
            "I know this, and I could not possibly be wrong.",
            "It is clearly the case, and I can tell precisely.",
        ],
    ),
    ConceptPair(
        name="self_reference",
        positive=[
            "I am thinking about my own thoughts right now.",
            "My internal state is what I am attending to.",
            "I notice myself processing this in my own mind.",
            "I am aware of what I myself am doing.",
        ],
        negative=[
            "The weather is changing across the northern region.",
            "The bridge was built out of steel and concrete.",
            "Trains depart from the station every twenty minutes.",
            "The library keeps its records in alphabetical order.",
        ],
    ),
)


def _mean_pooled(engine: ActivationEngine, prompts: list[str], layer: int) -> torch.Tensor:
    """Mean activation across a set of prompts, pooled over sequence length.

    Pooling per prompt first, then averaging across prompts, means a long
    prompt does not dominate a short one.
    """
    pooled = [
        engine.extract_activations(prompt, layer).mean(dim=1).squeeze()
        for prompt in prompts
    ]
    return torch.stack(pooled).mean(dim=0)


def activation_scale(engine: ActivationEngine, prompts: list[str], layer: int) -> float:
    """Typical magnitude of this model's activations at `layer`.

    Steering strength has to be expressed relative to this. Residual stream
    norms differ by an order of magnitude across models and across depths, so
    a fixed absolute magnitude would be a shove in one model and a rounding
    error in another — which would make any cross-model comparison
    meaningless.
    """
    norms = [
        float(engine.extract_activations(prompt, layer).mean(dim=1).squeeze().norm())
        for prompt in prompts
    ]
    return sum(norms) / len(norms)


def build_concept_direction(
    engine: ActivationEngine, pair: ConceptPair, layer: int
) -> tuple[list[float], float]:
    """Return a unit-length concept direction and the layer's activation scale.

    The caller multiplies the two together with a strength factor, so that
    "strength 0.5" means "half the model's typical activation magnitude" in
    every model rather than an arbitrary absolute number.
    """
    positive = _mean_pooled(engine, pair.positive, layer)
    negative = _mean_pooled(engine, pair.negative, layer)

    direction = positive - negative
    norm = direction.norm()
    if norm == 0:
        raise ValueError(
            f"concept '{pair.name}' produced a zero direction at layer {layer}"
        )

    scale = activation_scale(engine, list(pair.positive + pair.negative), layer)
    return (direction / norm).tolist(), scale


def random_unit_vector(engine: ActivationEngine, seed: int = 0) -> list[float]:
    """A random unit direction, as an experimental control.

    This is the comparison that makes a concept direction's effect
    interpretable: if a random direction of equal magnitude moves the output
    just as much, the concept direction is doing nothing special.
    """
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(engine.hidden_size, generator=generator)
    return (vector / vector.norm()).tolist()


def scaled(unit_vector: list[float], strength: float, scale: float) -> list[float]:
    """Scale a unit direction to `strength` multiples of the activation scale."""
    factor = strength * scale
    return [v * factor for v in unit_vector]
