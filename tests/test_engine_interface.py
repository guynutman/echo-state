import pytest

from echostate.engine import ActivationEngine


def test_engine_cannot_be_instantiated_without_an_implementation():
    with pytest.raises(TypeError):
        ActivationEngine()


def test_partial_implementation_is_rejected():
    class HalfEngine(ActivationEngine):
        @property
        def hidden_size(self) -> int:
            return 768

    with pytest.raises(TypeError):
        HalfEngine()


def test_full_implementation_satisfies_the_contract():
    class FakeEngine(ActivationEngine):
        @property
        def hidden_size(self) -> int:
            return 768

        @property
        def num_layers(self) -> int:
            return 12

        def extract_activations(
            self, prompt, target_layer, steering_vector=None, read_layer=None
        ):
            return None

        def generate_completion(self, prompt, max_new_tokens=100):
            return "baseline"

        def generate_steered_completion(
            self, prompt, target_layer, steering_vector, max_new_tokens=100
        ):
            return "steered"

        def compute_activation_divergence(self, baseline_acts, steered_acts):
            return 0.0

    engine = FakeEngine()

    assert engine.hidden_size == 768
    assert engine.num_layers == 12
    assert engine.generate_completion("hi") == "baseline"
    assert engine.generate_steered_completion("hi", 8, [0.0] * 768) == "steered"
