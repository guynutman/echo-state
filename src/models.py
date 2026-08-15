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


class ArtifactResult(BaseModel):
    """One row in the output CSV."""

    experiment_id: str
    is_control: bool
    raw_completion: str
    introspection_success: bool
    activation_divergence: float
    target_layer: int
    prompt: str


class ExperimentSuite(BaseModel):
    """Top-level collection of experiments in a JSON input file."""

    experiments: list[IntrospectionExperiment] = Field(default_factory=list)
