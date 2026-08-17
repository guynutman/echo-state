"""Orchestration: run experiments against an engine and produce result rows.

This module talks only to the ActivationEngine interface, never to torch or a
model directly. That is what lets it be tested against a fake engine without
downloading weights.
"""

from __future__ import annotations

import re
import sys
from difflib import SequenceMatcher

from echostate.engine import ActivationEngine
from echostate.models import ArtifactResult, ExperimentSuite, IntrospectionExperiment

# Words too common to count as evidence of anything.
_STOPWORDS = frozenset(
    """
    a an the and or of to in on is are am was were be been it its this that
    i you he she they we my your as at for with about
    """.split()
)


def _content_words(text: str) -> set[str]:
    """Lowercased word set, minus stopwords."""
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOPWORDS}


class Evaluator:
    """Runs experiments and turns them into ArtifactResult rows."""

    def __init__(self, engine: ActivationEngine, max_new_tokens: int = 60) -> None:
        self.engine = engine
        self.max_new_tokens = max_new_tokens
        # Many experiments share a prompt, and baseline generation is
        # deterministic, so the same baseline would otherwise be recomputed
        # dozens of times per model. Keyed by (prompt, target_layer,
        # read_layer) because activations depend on all three.
        self._baseline_cache: dict[tuple, tuple] = {}

    def _baseline(self, prompt: str, target_layer: int, read_layer: int):
        key = (prompt, target_layer, read_layer)
        if key not in self._baseline_cache:
            self._baseline_cache[key] = (
                self.engine.generate_completion(prompt, max_new_tokens=self.max_new_tokens),
                self.engine.extract_activations(
                    prompt, target_layer, read_layer=read_layer
                ),
            )
        return self._baseline_cache[key]

    def check_introspection_success(
        self, completion: str, expected_concept: str, prompt: str = ""
    ) -> bool:
        """Did the completion report the concept, without merely echoing the prompt?

        Words already present in the prompt are excluded. Without that, a
        model that simply continues "…inside your neural network:" by
        repeating "neural network" scores as a successful self-report, which
        measures autocomplete rather than introspection. That confound
        inflated every control row in the first version of this pipeline.
        """
        concept_words = _content_words(expected_concept)
        if prompt:
            concept_words -= _content_words(prompt)
        if not concept_words:
            return False

        return bool(concept_words & _content_words(completion))

    def compute_output_divergence(self, baseline: str, steered: str) -> float:
        """How much the text changed, in [0, 1]. 0.0 = identical.

        Pairs with activation_divergence: an intervention can move internals
        without changing behaviour, and the interesting cases are exactly
        where the two measures disagree.
        """
        if not baseline and not steered:
            return 0.0
        return 1.0 - SequenceMatcher(None, baseline, steered).ratio()

    def run_experiment(self, experiment: IntrospectionExperiment) -> list[ArtifactResult]:
        """Run one experiment: a baseline row, plus a steered row if steered.

        Both runs read activations at the same layer so their divergence is
        meaningful; a baseline row's divergence is 0.0 by definition, since it
        is the thing everything else is measured against.
        """
        read_layer = experiment.read_layer
        if read_layer is None:
            read_layer = self.engine.num_layers - 1

        model_name = getattr(self.engine, "model_name", "")

        baseline_completion, baseline_acts = self._baseline(
            experiment.prompt, experiment.target_layer, read_layer
        )

        results = [
            ArtifactResult(
                experiment_id=experiment.experiment_id,
                is_control=True,
                raw_completion=baseline_completion,
                introspection_success=self.check_introspection_success(
                    baseline_completion, experiment.expected_concept, experiment.prompt
                ),
                activation_divergence=0.0,
                output_divergence=0.0,
                target_layer=experiment.target_layer,
                prompt=experiment.prompt,
                model_name=model_name,
                concept_name=experiment.concept_name or "",
                steering_kind="none",
                steering_strength=0.0,
                read_layer=read_layer,
            )
        ]

        if experiment.steering_vector is None:
            return results

        steered_completion = self.engine.generate_steered_completion(
            experiment.prompt,
            experiment.target_layer,
            experiment.steering_vector,
            max_new_tokens=self.max_new_tokens,
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
                    steered_completion, experiment.expected_concept, experiment.prompt
                ),
                activation_divergence=self.engine.compute_activation_divergence(
                    baseline_acts, steered_acts
                ),
                output_divergence=self.compute_output_divergence(
                    baseline_completion, steered_completion
                ),
                target_layer=experiment.target_layer,
                prompt=experiment.prompt,
                model_name=model_name,
                concept_name=experiment.concept_name or "",
                steering_kind=experiment.steering_kind,
                steering_strength=experiment.steering_strength,
                read_layer=read_layer,
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
                f"  [{index}/{total}] {experiment.experiment_id}",
                file=sys.stderr,
                flush=True,
            )
            results.extend(self.run_experiment(experiment))

        return results
