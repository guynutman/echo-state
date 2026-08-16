"""Engine interface: the boundary between experiment logic and a model backend.

The evaluator never touches a model directly — it talks to an ActivationEngine.
That keeps the pipeline backend-agnostic: GPT-2 today, Llama or Mistral later,
with no change above this line.

torch is imported only for type checking so this module stays importable (and
testable) before the heavy dependencies are installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class ActivationEngine(ABC):
    """Read, steer, and compare a decoder-only model's internal state.

    Implementations own the model, the tokenizer, and the device. Every method
    must leave the model exactly as it found it: any hook registered during a
    call must be removed before the call returns, including on exception.
    """

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Width of the residual stream (768 for gpt2).

        Steering vectors must match this length; the evaluator uses it to
        generate or validate them.
        """

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer blocks, so a bad target_layer fails early."""

    @abstractmethod
    def extract_activations(
        self,
        prompt: str,
        target_layer: int,
        steering_vector: list[float] | None = None,
        read_layer: int | None = None,
    ) -> "torch.Tensor":
        """Run `prompt` and capture hidden states, optionally while steering.

        `target_layer` is where a steering vector is injected, if one is given.
        `read_layer` is where activations are captured; it defaults to
        `target_layer`.

        Pass `steering_vector` to measure the internal effect of an
        intervention. Note that reading at the injection layer is trivial —
        the result is exactly baseline + steering_vector — so to measure how
        far an intervention propagated, read at a later layer. Reading
        upstream of the injection shows no effect at all, since a layer
        cannot be influenced by one that runs after it.

        Returns shape (1, seq_len, hidden_size), detached and on CPU — the
        caller gets a plain tensor with no ties to the computation graph.
        """

    @abstractmethod
    def generate_completion(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate with no hooks attached. This is the baseline output.

        Returns only the newly generated text, with the prompt stripped.
        """

    @abstractmethod
    def generate_steered_completion(
        self,
        prompt: str,
        target_layer: int,
        steering_vector: list[float],
        max_new_tokens: int = 100,
    ) -> str:
        """Generate while adding `steering_vector` to `target_layer`'s output.

        The vector has length `hidden_size` and is broadcast across batch and
        sequence positions, so it perturbs every token as generation proceeds.
        Returns only the newly generated text.
        """

    @abstractmethod
    def compute_activation_divergence(
        self,
        baseline_acts: "torch.Tensor",
        steered_acts: "torch.Tensor",
    ) -> float:
        """Cosine distance between two activation tensors, in [0, 2].

        Mean-pools each tensor across sequence length first, so runs of
        different lengths remain comparable. 0.0 means "steering changed
        nothing at this layer".
        """
