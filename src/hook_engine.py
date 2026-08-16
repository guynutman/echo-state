"""GPT-2 implementation of ActivationEngine, using PyTorch forward hooks.

Requires: uv add torch transformers

A forward hook is a function PyTorch calls automatically whenever a module
produces output, handing you (module, input, output). Return None to observe
without changing anything; return a replacement to alter what the rest of the
network sees.

Two facts about GPT-2 that this whole file depends on:
  - model.transformer.h is the list of transformer blocks; h[i] is layer i.
  - a block's `output` is a TUPLE, not a tensor. Hidden states are output[0],
    shaped (batch, seq_len, hidden_size). Forgetting this is the single most
    common way this file breaks, and it usually breaks silently.
"""

from __future__ import annotations

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.engine import ActivationEngine


class HookEngine(ActivationEngine):
    """Runs GPT-2, reads its residual stream, and steers it mid-forward."""

    def __init__(self, model_name: str = "gpt2") -> None:
        """Load the model and tokenizer, pick a device, set eval mode."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name

    @property
    def hidden_size(self) -> int:
        """Width of the residual stream. Read from config: gpt2-medium is 1024."""
        return self.model.config.n_embd

    @property
    def num_layers(self) -> int:
        """Number of transformer blocks (12 for gpt2)."""
        return self.model.config.n_layer

    def _validate_layer(self, target_layer: int) -> None:
        """Fail loudly on an out-of-range layer, before any inference runs."""
        if not 0 <= target_layer < self.num_layers:
            raise ValueError(
                f"target_layer {target_layer} is out of range for "
                f"{self.model_name}: valid layers are 0..{self.num_layers - 1}"
            )

    def _validate_steering_vector(self, steering_vector: list[float]) -> None:
        """Fail loudly on a wrong-width vector.

        A width mismatch does NOT crash on its own: broadcasting quietly
        reinterprets it and you get plausible-looking garbage in your CSV
        with no traceback.
        """
        if len(steering_vector) != self.hidden_size:
            raise ValueError(
                f"steering_vector has length {len(steering_vector)}, "
                f"expected {self.hidden_size} for {self.model_name}"
            )

    def extract_activations(self, prompt: str, target_layer: int) -> torch.Tensor:
        """Capture hidden states leaving `target_layer`.

        Returns (1, seq_len, hidden_size), detached and on CPU.
        """
        self._validate_layer(target_layer)

        # The hook can't return the tensor — PyTorch calls it, not us — so it
        # closes over this list and appends into it instead.
        captured: list[torch.Tensor] = []

        def hook_fn(module, inputs, output):
            # detach: cut from the autograd graph. cpu: get it off the GPU.
            # clone: copy, since torch reuses the underlying buffer.
            captured.append(output[0].detach().cpu().clone())

        handle = self.model.transformer.h[target_layer].register_forward_hook(hook_fn)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # Return value is discarded: the model is run for the side
                # effect of the forward pass firing the hook.
                self.model(**inputs)
        finally:
            # Must be in a finally. A hook that survives an exception stays
            # attached, and the next "clean baseline" run is silently hooked.
            handle.remove()

        if not captured:
            raise RuntimeError(
                f"hook on layer {target_layer} never fired for {self.model_name}"
            )

        return captured[0]

    def generate_completion(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Baseline generation, with no hooks attached.

        Returns only the newly generated text, with the prompt stripped.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # Greedy, not sampled: a baseline/steered difference must be
                # attributable to the steering vector, not to the sampler.
                do_sample=False,
                # GPT-2 ships no pad token; without this transformers warns
                # and can pad with garbage.
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # generate() returns prompt + continuation. Slice past the prompt so
        # callers compare completions, not the prompt they already have.
        return self.tokenizer.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        )

    def generate_steered_completion(
        self,
        prompt: str,
        target_layer: int,
        steering_vector: list[float],
        max_new_tokens: int = 100,
    ) -> str:
        """Generate with `steering_vector` added to `target_layer`'s output.

        The hook fires on every forward pass, so every generated token is
        steered, not just the first. Returns only the newly generated text.
        """
        self._validate_layer(target_layer)
        self._validate_steering_vector(steering_vector)

        # Built once, outside the hook: building it inside would re-allocate
        # on every generated token.
        sv = torch.tensor(steering_vector, dtype=torch.float32, device=self.device)

        def hook_fn(module, inputs, output):
            hidden = output[0] + sv  # (d,) broadcasts over (batch, seq, d)
            # Rebuild the tuple. Returning a bare tensor breaks the next block,
            # and output[1:] carries the KV cache that generation depends on.
            return (hidden,) + output[1:]

        handle = self.model.transformer.h[target_layer].register_forward_hook(hook_fn)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            handle.remove()

        return self.tokenizer.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        )

    def compute_activation_divergence(
        self,
        baseline_acts: torch.Tensor,
        steered_acts: torch.Tensor,
    ) -> float:
        """Cosine distance between two runs' activations, in [0, 2].

        0.0 means steering changed nothing at this layer.
        """
        # Mean-pool across sequence length so runs of different token counts
        # stay comparable: (1, seq, d) -> (d,)
        b_pooled = baseline_acts.mean(dim=1).squeeze()
        s_pooled = steered_acts.mean(dim=1).squeeze()

        # Cosine compares direction only, ignoring magnitude.
        cos_sim = torch.nn.functional.cosine_similarity(b_pooled, s_pooled, dim=0)

        return float((1 - cos_sim).item())
