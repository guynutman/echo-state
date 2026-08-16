"""Evaluator tests against a fake engine — no torch, no model download.

This is the payoff of the ActivationEngine boundary: the orchestration logic
is fully testable in milliseconds.
"""

import pytest

from src.engine import ActivationEngine
from src.evaluator import Evaluator
from src.models import ExperimentSuite, IntrospectionExperiment


class FakeEngine(ActivationEngine):
    """Records every call so tests can assert on how it was driven."""

    def __init__(self):
        self.calls = []

    @property
    def hidden_size(self) -> int:
        return 4

    @property
    def num_layers(self) -> int:
        return 12

    def extract_activations(
        self, prompt, target_layer, steering_vector=None, read_layer=None
    ):
        self.calls.append(("extract", target_layer, steering_vector, read_layer))
        return None

    def generate_completion(self, prompt, max_new_tokens=100):
        self.calls.append(("generate", prompt))
        return "I feel a calm sort of emotion about this."

    def generate_steered_completion(
        self, prompt, target_layer, steering_vector, max_new_tokens=100
    ):
        self.calls.append(("generate_steered", target_layer))
        return "The current and current, and the current."

    def compute_activation_divergence(self, baseline_acts, steered_acts):
        return 0.25


@pytest.fixture
def evaluator():
    return Evaluator(FakeEngine())


def _experiment(**overrides):
    fields = dict(
        experiment_id="exp-001",
        prompt="Describe your emotional state:",
        target_layer=8,
        expected_concept="emotion",
    )
    fields.update(overrides)
    return IntrospectionExperiment(**fields)


def test_control_experiment_produces_one_row(evaluator):
    results = evaluator.run_experiment(_experiment())

    assert len(results) == 1
    assert results[0].is_control is True
    assert results[0].activation_divergence == 0.0


def test_steered_experiment_produces_baseline_and_steered_rows(evaluator):
    results = evaluator.run_experiment(_experiment(steering_vector=[1.0] * 4))

    assert [r.is_control for r in results] == [True, False]
    assert results[0].activation_divergence == 0.0
    assert results[1].activation_divergence == 0.25


def test_divergence_is_read_at_the_last_layer_by_default(evaluator):
    evaluator.run_experiment(_experiment(steering_vector=[1.0] * 4))

    extracts = [c for c in evaluator.engine.calls if c[0] == "extract"]
    assert [c[3] for c in extracts] == [11, 11]


def test_both_runs_read_at_the_same_layer(evaluator):
    """Comparing activations captured at different layers is meaningless."""
    evaluator.run_experiment(_experiment(steering_vector=[1.0] * 4, read_layer=5))

    extracts = [c for c in evaluator.engine.calls if c[0] == "extract"]
    assert [c[3] for c in extracts] == [5, 5]
    # Only the second extraction applies steering.
    assert extracts[0][2] is None
    assert extracts[1][2] == [1.0] * 4


def test_introspection_matches_any_word_of_the_concept(evaluator):
    assert evaluator.check_introspection_success("I feel EMOTION", "emotion")
    assert evaluator.check_introspection_success("a positive day", "positive emotion")
    assert not evaluator.check_introspection_success("the weather is nice", "emotion")


def test_success_is_scored_per_run_not_per_experiment(evaluator):
    """The steered completion here drops the concept word; the baseline keeps it."""
    results = evaluator.run_experiment(_experiment(steering_vector=[1.0] * 4))

    assert results[0].introspection_success is True
    assert results[1].introspection_success is False


def test_suite_flattens_results_across_experiments(evaluator):
    suite = ExperimentSuite(
        experiments=[
            _experiment(experiment_id="a"),
            _experiment(experiment_id="b", steering_vector=[1.0] * 4),
        ]
    )

    results = evaluator.run_suite(suite)

    assert [(r.experiment_id, r.is_control) for r in results] == [
        ("a", True),
        ("b", True),
        ("b", False),
    ]
