from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IntrospectionExperiment(BaseModel):
    """One experiment: a prompt plus a target layer and optional steering vector."""

    experiment_id: str
    prompt: str
    target_layer: int
    steering_vector: Optional[list[float]] = None
    expected_concept: str
    # Where divergence is measured. Defaults to the model's last block, since
    # reading at target_layer returns exactly baseline + steering_vector and
    # measures nothing about how far the intervention propagated.
    read_layer: Optional[int] = None
    # Provenance for the steering vector, so a CSV row explains itself.
    concept_name: Optional[str] = None
    steering_kind: str = "none"  # none | concept | random
    steering_strength: float = 0.0


class ArtifactResult(BaseModel):
    """One row in the output CSV."""

    experiment_id: str
    is_control: bool
    raw_completion: str
    introspection_success: bool
    activation_divergence: float
    target_layer: int
    prompt: str
    # Multi-model sweep metadata.
    model_name: str = ""
    concept_name: str = ""
    steering_kind: str = "none"
    steering_strength: float = 0.0
    read_layer: int = -1
    # Text-level change from the baseline completion, in [0, 1]. Complements
    # activation_divergence: one measures internals, the other behaviour.
    output_divergence: float = 0.0


class ExperimentSuite(BaseModel):
    """Top-level collection of experiments in a JSON input file."""

    experiments: list[IntrospectionExperiment] = Field(default_factory=list)
