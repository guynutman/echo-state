"""Orchestration: run experiments against an engine and produce result rows.

This module talks only to the ActivationEngine interface, never to torch or a
model directly. That is what lets it be tested against a fake engine without
downloading weights.
"""

from __future__ import annotations

import sys

from src.engine import ActivationEngine
from src.models import ArtifactResult, ExperimentSuite, IntrospectionExperiment


class Evaluator:
    """Runs experiments and turns them into ArtifactResult rows."""

    def __init__(self, engine: ActivationEngine) -> None:
        self.engine = engine

    def check_introspection_success(
        self, completion: str, expected_concept: str
    ) -> bool:
        """Does the completion mention the concept it was expected to report?

        V1: case-insensitive match on any word of the concept. "positive
        emotion" counts as a hit if the completion contains either "positive"
        or "emotion".

        This is a blunt instrument and its weakness belongs in the writeup: it
        detects vocabulary, not understanding, and cannot tell a genuine
        self-report from an incidental use of the word.
        """
        haystack = completion.lower()
        return any(word in haystack for word in expected_concept.lower().split())

    def run_experiment(
        self, experiment: IntrospectionExperiment
    ) -> list[ArtifactResult]:
        """Run one experiment: a baseline row, plus a steered row if steered.

        Both runs read activations at the same layer so their divergence is
        meaningful; a baseline row's divergence is 0.0 by definition, since it
        is the thing everything else is measured against.
        """
        read_layer = experiment.read_layer
        if read_layer is None:
            read_layer = self.engine.num_layers - 1

        baseline_completion = self.engine.generate_completion(experiment.prompt)
        baseline_acts = self.engine.extract_activations(
            experiment.prompt, experiment.target_layer, read_layer=read_layer
        )

        results = [
            ArtifactResult(
                experiment_id=experiment.experiment_id,
                is_control=True,
                raw_completion=baseline_completion,
                introspection_success=self.check_introspection_success(
                    baseline_completion, experiment.expected_concept
                ),
                activation_divergence=0.0,
                target_layer=experiment.target_layer,
                prompt=experiment.prompt,
            )
        ]

        if experiment.steering_vector is None:
            return results

        steered_completion = self.engine.generate_steered_completion(
            experiment.prompt, experiment.target_layer, experiment.steering_vector
        )
        steered_acts = self.engine.extract_activations(
            experiment.prompt,
            experiment.target_layer,
            steering_vector=experiment.steering_vector,
            read_layer=read_layer,
        )

        results.append(
            ArtifactResult(
                experiment_id=experiment.experiment_id,
                is_control=False,
                raw_completion=steered_completion,
                introspection_success=self.check_introspection_success(
                    steered_completion, experiment.expected_concept
                ),
                activation_divergence=self.engine.compute_activation_divergence(
                    baseline_acts, steered_acts
                ),
                target_layer=experiment.target_layer,
                prompt=experiment.prompt,
            )
        )

        return results

    def run_suite(self, suite: ExperimentSuite) -> list[ArtifactResult]:
        """Run every experiment, reporting progress on stderr.

        Progress goes to stderr so stdout stays clean for piped output.
        """
        results: list[ArtifactResult] = []
        total = len(suite.experiments)

        for index, experiment in enumerate(suite.experiments, start=1):
            print(
                f"[{index}/{total}] {experiment.experiment_id}",
                file=sys.stderr,
                flush=True,
            )
            results.extend(self.run_experiment(experiment))

        return results
