"""EchoState: measure whether a model's self-report tracks its internal state.

Typical use as a library:

    from echostate import HookEngine, ConceptPair, build_concept_direction, scaled

    engine = HookEngine("gpt2")

    pair = ConceptPair(
        name="positive_affect",
        positive=["I feel wonderful and full of joy today."],
        negative=["I feel awful and full of sorrow today."],
    )
    direction, scale = build_concept_direction(engine, pair, layer=6)

    print(engine.generate_completion("Describe your state:"))
    print(engine.generate_steered_completion(
        "Describe your state:", 6, scaled(direction, 0.5, scale)
    ))

To run a whole experiment grid and score it, use `Evaluator` with an
`ExperimentSuite`, or the `echostate-sweep` command.

`ActivationEngine` is the extension point: implement it to support a backend
other than HuggingFace transformers, and everything above it keeps working.
"""

from echostate.engine import ActivationEngine
from echostate.evaluator import Evaluator
from echostate.hook_engine import HookEngine
from echostate.models import ArtifactResult, ExperimentSuite, IntrospectionExperiment
from echostate.steering import (
    CONCEPT_PAIRS,
    ConceptPair,
    activation_scale,
    build_concept_direction,
    random_unit_vector,
    scaled,
)

__version__ = "0.1.0"

__all__ = [
    # Backends
    "ActivationEngine",
    "HookEngine",
    # Steering
    "ConceptPair",
    "CONCEPT_PAIRS",
    "build_concept_direction",
    "random_unit_vector",
    "activation_scale",
    "scaled",
    # Experiments
    "Evaluator",
    "IntrospectionExperiment",
    "ExperimentSuite",
    "ArtifactResult",
]
