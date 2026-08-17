"""Hook-based ActivationEngine for any HuggingFace causal language model.

Requires: uv add torch transformers

A forward hook is a function PyTorch calls automatically whenever a module
produces output, handing you (module, input, output). Return None to observe
without changing anything; return a replacement to alter what the rest of the
network sees.

Two things this file has to paper over, both of which break silently:

  - WHERE the transformer blocks live differs by architecture. GPT-2 and
    GPT-Neo use model.transformer.h, Llama/Qwen use model.model.layers,
    Pythia uses model.gpt_neox.layers, OPT uses model.model.decoder.layers.
    _resolve_blocks finds whichever one exists.

  - WHAT a block hands the hook as `output` depends on the transformers
    version. In 4.x it is a tuple of (hidden_states, kv_cache, ...); in 5.x it
    is a bare Tensor. Assuming a tuple under 5.x makes output[0] index the
    batch dimension instead, returning a wrongly shaped tensor without
    raising. Use the helpers below rather than indexing a block output.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.engine import ActivationEngine

# Known locations of the transformer block list, by architecture family.
_BLOCK_PATHS = (
    "transformer.h",            # GPT-2, GPT-Neo, GPT-J
    "model.layers",             # Llama, Qwen, Mistral, Gemma
    "gpt_neox.layers",          # Pythia, GPT-NeoX
    "model.decoder.layers",     # OPT
    "transformer.blocks",       # MPT
)


def _hidden_states(output):
    """Pull hidden states out of whatever shape a block returned."""
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden_states(output, hidden):
    """Rebuild a block output with `hidden` swapped in, preserving any extras.

    Under 4.x the extras are the KV cache that generation depends on, so they
    must be carried through rather than dropped.
    """
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


def _resolve_blocks(model):
    """Find the model's list of transformer blocks, whatever it is called.

    Raises rather than guessing: a wrong module would hook something that is
    not the residual stream and produce plausible, meaningless numbers.
    """
    for path in _BLOCK_PATHS:
        node = model
        for attribute in path.split("."):
            node = getattr(node, attribute, None)
            if node is None:
                break
        if node is not None and len(node) > 0:
            return node

    raise ValueError(
        f"could not locate transformer blocks on {type(model).__name__}; "
        f"tried {', '.join(_BLOCK_PATHS)}"
    )


class HookEngine(ActivationEngine):
    """Runs a causal LM, reads its residual stream, and steers it mid-forward."""

    def __init__(
        self,
        model_name: str = "gpt2",
        no_repeat_ngram_size: int = 3,
    ) -> None:
        """Load the model and tokenizer, pick a device, set eval mode.

        `no_repeat_ngram_size` forbids repeating any n-gram of that length.
        Greedy decoding on a small model falls into attractor loops — the
        context that produced a phrase makes that phrase most likely again,
        forever. This breaks the cycle while staying fully deterministic,
        which sampling would not.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.no_repeat_ngram_size = no_repeat_ngram_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.blocks = _resolve_blocks(self.model)

        # Not every tokenizer ships a pad token; generation wants one.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def hidden_size(self) -> int:
        """Width of the residual stream, under whichever name the config uses."""
        config = self.model.config
        for attribute in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, attribute, None)
            if value is not None:
                return value
        raise ValueError(f"could not determine hidden size for {self.model_name}")

    @property
    def num_layers(self) -> int:
        """Counted from the resolved blocks, not from config, so the two cannot drift."""
        return len(self.blocks)

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

    def _make_steering_hook(self, steering_vector: list[float]):
        """Build a forward hook that adds `steering_vector` to a block's output.

        The tensor is built here, once, rather than inside the hook: the hook
        fires on every forward pass, so allocating inside it would re-allocate
        on every generated token.
        """
        sv = torch.tensor(steering_vector, dtype=torch.float32, device=self.device)

        def hook_fn(module, inputs, output):
            hidden = _hidden_states(output)
            # Match the model's dtype. Some checkpoints (pythia-410m) load in
            # float16; adding a float32 vector silently promotes the hidden
            # states, and the next layer norm then fails on mixed dtypes.
            # (d,) broadcasts over (batch, seq, d), so this works both on the
            # first pass and on the single-token passes that follow it.
            return _replace_hidden_states(output, hidden + sv.to(hidden.dtype))

        return hook_fn

    def _generation_kwargs(self, max_new_tokens: int) -> dict:
        return dict(
            max_new_tokens=max_new_tokens,
            # Greedy, not sampled: a baseline/steered difference must be
            # attributable to the steering vector, not to the sampler.
            do_sample=False,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def extract_activations(
        self,
        prompt: str,
        target_layer: int,
        steering_vector: list[float] | None = None,
        read_layer: int | None = None,
    ) -> torch.Tensor:
        """Capture hidden states at `read_layer`, optionally steering at `target_layer`.

        Returns (1, seq_len, hidden_size), detached and on CPU.
        """
        self._validate_layer(target_layer)
        if read_layer is None:
            read_layer = target_layer
        self._validate_layer(read_layer)
        if steering_vector is not None:
            self._validate_steering_vector(steering_vector)

        # The hook can't return the tensor — PyTorch calls it, not us — so it
        # closes over this list and appends into it instead.
        captured: list[torch.Tensor] = []

        def read_hook(module, inputs, output):
            # detach: cut from the autograd graph. cpu: get it off the GPU.
            # clone: copy, since torch reuses the underlying buffer.
            captured.append(_hidden_states(output).detach().cpu().clone())

        handles = []
        try:
            # Register steering FIRST. When a forward hook returns a value,
            # PyTorch passes that value to later hooks on the same module, so
            # this ordering is what lets read_layer == target_layer see the
            # steered state rather than the original.
            if steering_vector is not None:
                handles.append(
                    self.blocks[target_layer].register_forward_hook(
                        self._make_steering_hook(steering_vector)
                    )
                )
            handles.append(self.blocks[read_layer].register_forward_hook(read_hook))

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # Return value is discarded: the model is run for the side
                # effect of the forward pass firing the hooks.
                self.model(**inputs)
        finally:
            # Must be in a finally. A hook that survives an exception stays
            # attached, and the next "clean baseline" run is silently hooked.
            for handle in handles:
                handle.remove()

        if not captured:
            raise RuntimeError(
                f"hook on layer {read_layer} never fired for {self.model_name}"
            )

        return captured[0]

    def extract_all_layers(
        self,
        prompt: str,
        target_layer: int,
        steering_vector: list[float] | None = None,
    ) -> list[torch.Tensor]:
        """Capture activations at EVERY layer in a single forward pass.

        Used to trace how far an intervention propagates. Doing this with one
        pass rather than N keeps a multi-model sweep tractable.
        """
        self._validate_layer(target_layer)
        if steering_vector is not None:
            self._validate_steering_vector(steering_vector)

        captured: dict[int, torch.Tensor] = {}

        def make_read_hook(index: int):
            def read_hook(module, inputs, output):
                captured[index] = _hidden_states(output).detach().cpu().clone()

            return read_hook

        handles = []
        try:
            if steering_vector is not None:
                handles.append(
                    self.blocks[target_layer].register_forward_hook(
                        self._make_steering_hook(steering_vector)
                    )
                )
            for index in range(self.num_layers):
                handles.append(
                    self.blocks[index].register_forward_hook(make_read_hook(index))
                )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for handle in handles:
                handle.remove()

        return [captured[index] for index in range(self.num_layers)]

    def generate_completion(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Baseline generation, with no hooks attached.

        Returns only the newly generated text, with the prompt stripped.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, **self._generation_kwargs(max_new_tokens)
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

        handle = self.blocks[target_layer].register_forward_hook(
            self._make_steering_hook(steering_vector)
        )
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, **self._generation_kwargs(max_new_tokens)
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
