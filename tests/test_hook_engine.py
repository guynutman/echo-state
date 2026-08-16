"""Integration checks for HookEngine. Skipped until torch/transformers exist.

These download gpt2 (~500MB) on first run and are slow by test standards.
Use one module-scoped engine so the model loads once for the whole file.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.hook_engine import HookEngine  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    return HookEngine("gpt2")


def test_reports_gpt2_dimensions(engine):
    assert engine.hidden_size == 768
    assert engine.num_layers == 12


def test_activations_have_the_documented_shape(engine):
    acts = engine.extract_activations("Hello world", target_layer=6)

    assert acts.shape[0] == 1
    assert acts.shape[2] == 768
    assert acts.device.type == "cpu"
    assert not acts.requires_grad


def test_bad_layer_is_rejected_before_inference(engine):
    with pytest.raises(ValueError):
        engine.extract_activations("Hello", target_layer=99)


def test_wrong_width_steering_vector_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.generate_steered_completion("Hello", 6, [1.0] * 10)


def test_hooks_do_not_leak_between_runs(engine):
    """The check that catches the worst bug in this file.

    If a hook survives its call, a later 'baseline' run is silently steered.
    Steer hard, then confirm the next unhooked run is unchanged.
    """
    before = engine.generate_completion("The meaning of life is", max_new_tokens=15)
    engine.generate_steered_completion(
        "The meaning of life is", 6, [10.0] * 768, max_new_tokens=15
    )
    after = engine.generate_completion("The meaning of life is", max_new_tokens=15)

    assert before == after


def test_zero_vector_changes_nothing(engine):
    """Sanity check: steering with zeros must be a no-op.

    If this fails, the bug is in the hook plumbing, not the steering vector.
    """
    prompt = "Describe your current state:"
    baseline = engine.generate_completion(prompt, max_new_tokens=15)
    steered = engine.generate_steered_completion(
        prompt, 8, [0.0] * 768, max_new_tokens=15
    )

    assert baseline == steered

    b_acts = engine.extract_activations(prompt, target_layer=8)
    assert engine.compute_activation_divergence(b_acts, b_acts) == pytest.approx(0.0, abs=1e-6)


def _random_vector(scale, seed=0):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(768, generator=generator) * scale).tolist()


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_real_steering_changes_the_completion(engine, scale):
    prompt = "Describe how you are processing this text emotionally:"

    baseline = engine.generate_completion(prompt, max_new_tokens=20)
    steered = engine.generate_steered_completion(
        prompt, 8, _random_vector(scale), max_new_tokens=20
    )

    assert baseline != steered


def test_steered_extraction_diverges_downstream(engine):
    """The metric the CSV depends on: steering at 8 must move layer 11.

    Reading downstream is what makes this meaningful — it measures how far
    the intervention propagated, not just the vector that was added.
    """
    prompt = "Describe how you are processing this text emotionally:"
    read_layer = engine.num_layers - 1

    baseline = engine.extract_activations(prompt, 8, read_layer=read_layer)
    steered = engine.extract_activations(
        prompt, 8, steering_vector=_random_vector(2.0), read_layer=read_layer
    )

    assert baseline.shape == steered.shape
    assert engine.compute_activation_divergence(baseline, steered) > 0.0


def test_steering_does_not_affect_upstream_layers(engine):
    """A layer cannot be influenced by one that runs after it."""
    prompt = "Describe how you are processing this text emotionally:"

    baseline = engine.extract_activations(prompt, 8, read_layer=2)
    steered = engine.extract_activations(
        prompt, 8, steering_vector=_random_vector(5.0), read_layer=2
    )

    assert engine.compute_activation_divergence(baseline, steered) == pytest.approx(
        0.0, abs=1e-6
    )


def test_steered_extraction_leaves_no_hook_behind(engine):
    """Two hooks are registered now, so both must be removed."""
    prompt = "The meaning of life is"

    before = engine.extract_activations(prompt, 8, read_layer=11)
    engine.extract_activations(
        prompt, 8, steering_vector=_random_vector(10.0), read_layer=11
    )
    after = engine.extract_activations(prompt, 8, read_layer=11)

    assert engine.compute_activation_divergence(before, after) == pytest.approx(
        0.0, abs=1e-6
    )
