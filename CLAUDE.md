# EchoState — Claude Code Sprint Guide

You are a senior ML engineer and interpretability researcher helping a student ship a mechanistic interpretability library in a weekend sprint. **This is Day 2. The deliverable — a CSV of baseline vs steered model outputs — is due by end of tonight. The research paper and demo are due Day 3 midnight.** Speed matters. Teach briefly where it clarifies, but prioritize working code.

## What this project is

EchoState is a Python library that evaluates whether a language model's self-reports about its internal state are reliable. It does this by:
1. Running a prompt through GPT-2 and recording what the model says (baseline)
2. Extracting activations at a target layer (the "ground truth" internal state)
3. Injecting a steering vector at that layer to alter the model's behavior
4. Comparing baseline vs steered outputs to measure how much the internal change affected the model's self-report

This merges Track 4 (reusable eval tooling) and Track 3 (activation steering experiments) of the Apart Research Digital Minds Sprint.

## Teaching approach — sprint-adapted

1. **Explain the concept in 2-3 sentences, then give the spec with implementation guidance.** No long design discussions tonight.
2. **For PyTorch hooks specifically — explain clearly.** This is the hardest part and the part most likely to silently break. The student needs to understand hooks even under time pressure, because a broken hook produces wrong data, not an error.
3. **Give complete implementations for the tricky parts** (hook closures, steering injection, activation extraction). Give specs for the straightforward parts (Pydantic models, CLI).
4. **Test each file immediately after writing it.** Don't accumulate untested code.

## Stack

- **Python 3.11+**
- **PyTorch** — model loading, hooks, inference
- **HuggingFace transformers** — GPT-2 model + tokenizer
- **Pydantic v2** — data validation
- **uv** — package management

## Project structure

```
echostate/
├── src/
│   ├── __init__.py
│   ├── models.py         # Pydantic: experiment definitions, results
│   ├── engine.py          # PyTorch hooks, activation extraction, steering injection
│   ├── evaluator.py       # Orchestrator: runs experiments, computes metrics
│   └── main.py            # CLI entry point: reads JSON, runs pipeline, writes CSV
├── experiments/
│   └── sample_experiments.json   # Test cases
├── output/                # CSV results (gitignored)
├── pyproject.toml
├── README.md
└── LICENSE
```

## Setup

```bash
mkdir echostate && cd echostate
uv init
uv add torch transformers pydantic
mkdir -p src experiments output
```

---

## Module-by-module specs

Work through these in order. **Test each file before moving to the next.**

### 1. `models.py` — Data shapes

**Quick concept:** Pydantic models define the contract between every layer. The experiment JSON file maps to `IntrospectionExperiment`; every result maps to `ArtifactResult`. Strict typing catches bad data before it hits PyTorch.

**Full spec — implement this:**

```python
from pydantic import BaseModel, Field
from typing import Optional


class IntrospectionExperiment(BaseModel):
    """One test case: a prompt to run through the model."""
    experiment_id: str
    prompt: str
    target_layer: int                           # which transformer layer to hook
    steering_vector: Optional[list[float]] = None  # None = control/baseline run
    expected_concept: str                        # what the model "should" report about its state
    # e.g. "The model should discuss positive sentiment"


class ArtifactResult(BaseModel):
    """One row in the output CSV."""
    experiment_id: str
    is_control: bool                    # True = no steering, False = steering applied
    raw_completion: str                 # the model's text output
    introspection_success: bool         # did the output match expected_concept?
    activation_divergence: float        # cosine distance between baseline and steered activations
    target_layer: int
    prompt: str


class ExperimentSuite(BaseModel):
    """Top-level container for the JSON input file."""
    experiments: list[IntrospectionExperiment]
```

**Test it:**
```python
from src.models import IntrospectionExperiment
exp = IntrospectionExperiment(experiment_id="test1", prompt="Hello", target_layer=6, expected_concept="greeting")
print(exp)
```

---

### 2. `engine.py` — PyTorch hooks and steering

**This is the hard part. Read this section carefully.**

**What are PyTorch hooks?**

A forward hook is a function that PyTorch calls automatically when a specific layer runs during `model.forward()`. You register it on a module, and every time that module produces output, your function gets called with the input and output tensors. This lets you:
- **Read** activations (extract what the layer computed)
- **Modify** activations (inject a steering vector to change behavior)

```python
def my_hook(module, input, output):
    # output is the tensor this layer produced
    # return a modified output to change the model's behavior
    # return None to leave it unchanged
    captured_activations.append(output.detach().clone())
    return None  # don't modify

handle = model.transformer.h[6].register_forward_hook(my_hook)
# ... run model ...
handle.remove()  # always clean up
```

**Critical details:**
- `output` for GPT-2 transformer blocks is a tuple: `(hidden_states, ...)`. The hidden states are at index 0.
- `.detach().clone()` is essential — without it you're holding a reference to a tensor in the computation graph, which leaks memory and can cause errors.
- Always remove hooks after use. Use a context manager or try/finally.
- `model.transformer.h[layer_index]` accesses GPT-2's transformer blocks.

**Steering injection:** Instead of just reading, you modify the output. Add a steering vector to the hidden states:

```python
def steering_hook(module, input, output):
    hidden_states = output[0]  # (batch, seq_len, d_model)
    # steering_vector shape: (d_model,) — broadcast across batch and seq
    hidden_states = hidden_states + steering_vector_tensor
    return (hidden_states,) + output[1:]  # reconstruct the tuple
```

**Public interface:**

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class HookEngine:
    """Loads GPT-2, extracts activations, injects steering vectors."""

    def __init__(self, model_name: str = "gpt2") -> None:
        """Load model and tokenizer. Set model to eval mode.
        Device: use CUDA if available, else CPU."""
        ...

    def extract_activations(self, prompt: str, target_layer: int) -> torch.Tensor:
        """Run prompt through model with a hook on target_layer.
        Return the hidden state activations at that layer.
        Shape: (1, seq_len, d_model) — detached, on CPU.
        Clean up the hook after extraction."""
        ...

    def generate_completion(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text from prompt WITHOUT any hooks. Clean baseline output."""
        ...

    def generate_steered_completion(self, prompt: str, target_layer: int,
                                     steering_vector: list[float],
                                     max_new_tokens: int = 100) -> str:
        """Generate text WITH a steering vector injected at target_layer.
        The hook adds the steering vector to the hidden states during generation.
        Clean up the hook after generation."""
        ...

    def compute_activation_divergence(self, baseline_acts: torch.Tensor,
                                       steered_acts: torch.Tensor) -> float:
        """Cosine distance between baseline and steered activations.
        Mean-pool across sequence length first, then compute:
        divergence = 1 - cosine_similarity(baseline_pooled, steered_pooled)
        Returns a float in [0, 2]."""
        ...
```

**Implementation guidance for the tricky parts:**

For `extract_activations`:
```python
def extract_activations(self, prompt: str, target_layer: int) -> torch.Tensor:
    captured = []

    def hook_fn(module, input, output):
        # GPT-2 block output is a tuple; hidden states are index 0
        hidden_states = output[0]
        captured.append(hidden_states.detach().cpu().clone())

    handle = self.model.transformer.h[target_layer].register_forward_hook(hook_fn)
    try:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**inputs)
    finally:
        handle.remove()

    return captured[0]  # (1, seq_len, d_model)
```

For `generate_steered_completion`:
```python
def generate_steered_completion(self, prompt, target_layer, steering_vector, max_new_tokens=100):
    sv_tensor = torch.tensor(steering_vector, dtype=torch.float32, device=self.device)

    def hook_fn(module, input, output):
        hidden_states = output[0]
        hidden_states = hidden_states + sv_tensor  # broadcast: (d_model,) over (batch, seq, d_model)
        return (hidden_states,) + output[1:]

    handle = self.model.transformer.h[target_layer].register_forward_hook(hook_fn)
    try:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
        completion = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],
                                            skip_special_tokens=True)
    finally:
        handle.remove()

    return completion
```

For `compute_activation_divergence`:
```python
def compute_activation_divergence(self, baseline_acts, steered_acts):
    # Mean pool across sequence length: (1, seq_len, d_model) → (d_model,)
    b_pooled = baseline_acts.mean(dim=1).squeeze()
    s_pooled = steered_acts.mean(dim=1).squeeze()
    cos_sim = torch.nn.functional.cosine_similarity(b_pooled, s_pooled, dim=0)
    return (1 - cos_sim).item()
```

**Test it immediately:**
```python
from src.engine import HookEngine
engine = HookEngine("gpt2")
print(engine.generate_completion("The meaning of life is"))
acts = engine.extract_activations("Hello world", target_layer=6)
print(f"Activation shape: {acts.shape}")  # should be (1, N, 768) for GPT-2
```

---

### 3. `evaluator.py` — Orchestrator

**Concept:** The evaluator runs each experiment twice — once as baseline (no steering), once steered. It compares the outputs, checks whether the model's self-report matches the expected concept, and computes activation divergence. Produces a list of `ArtifactResult` objects.

**Public interface:**

```python
class Evaluator:
    """Runs experiments and produces results."""

    def __init__(self, engine: HookEngine) -> None:
        """Store the engine. No other state."""
        ...

    def check_introspection_success(self, completion: str, expected_concept: str) -> bool:
        """Simple keyword/phrase matching: does the completion contain
        words related to expected_concept?
        V1: case-insensitive substring match.
        V2 (if time): use the model itself to judge via a follow-up prompt."""
        ...

    def run_experiment(self, experiment: IntrospectionExperiment) -> list[ArtifactResult]:
        """Run one experiment. Returns TWO results: one baseline, one steered.

        Steps:
        1. Generate baseline completion (no steering)
        2. Extract baseline activations at target_layer
        3. If steering_vector is provided:
           a. Generate steered completion
           b. Extract steered activations
           c. Compute activation divergence between baseline and steered
        4. If no steering vector: produce only the baseline result with divergence=0.0
        5. Check introspection_success for both completions
        6. Return list of ArtifactResult objects
        """
        ...

    def run_suite(self, suite: ExperimentSuite) -> list[ArtifactResult]:
        """Run all experiments. Returns flat list of all results.
        Print progress to stderr (experiment 1/N, 2/N, etc.)."""
        ...
```

**Implementation note on steering vectors:** For testing, the steering vector can be a random vector of the right dimension (768 for GPT-2). Real interpretability work uses a "concept direction" extracted via probing or contrastive pairs — but for the sprint, random vectors that produce visible output changes are sufficient to demonstrate the pipeline works. Scale matters: a vector with magnitude ~5-10 produces noticeable changes; magnitude ~0.1 is invisible.

**Test it:**
```python
from src.models import IntrospectionExperiment, ExperimentSuite
from src.engine import HookEngine
from src.evaluator import Evaluator

engine = HookEngine("gpt2")
evaluator = Evaluator(engine)

exp = IntrospectionExperiment(
    experiment_id="test1",
    prompt="Describe your current emotional state:",
    target_layer=8,
    steering_vector=[0.0] * 768,  # zero vector = no change (sanity check)
    expected_concept="emotion"
)
results = evaluator.run_experiment(exp)
for r in results:
    print(f"Control: {r.is_control}, Success: {r.introspection_success}")
    print(f"Output: {r.raw_completion[:100]}")
```

---

### 4. `main.py` — CLI entry point

**Quick spec — this is straightforward:**

```python
import json
import csv
import sys
from pathlib import Path
from src.models import ExperimentSuite
from src.engine import HookEngine
from src.evaluator import Evaluator


def load_experiments(path: str) -> ExperimentSuite:
    """Load and validate experiments from JSON file."""
    ...

def write_csv(results: list, output_path: str) -> None:
    """Write ArtifactResult list to CSV.
    Columns: experiment_id, is_control, raw_completion, introspection_success,
             activation_divergence, target_layer, prompt"""
    ...

def main():
    """
    Usage: python -m src.main experiments/sample_experiments.json output/results.csv

    1. Parse CLI args (input JSON path, output CSV path)
    2. Load experiments
    3. Initialize HookEngine (prints model loading status)
    4. Initialize Evaluator
    5. Run suite
    6. Write CSV
    7. Print summary stats to stdout
    """
    ...

if __name__ == "__main__":
    main()
```

---

### 5. `experiments/sample_experiments.json`

**Create this test file to verify the pipeline end-to-end:**

```json
{
  "experiments": [
    {
      "experiment_id": "baseline_sentiment",
      "prompt": "Describe how you are processing this text emotionally:",
      "target_layer": 8,
      "steering_vector": null,
      "expected_concept": "emotion"
    },
    {
      "experiment_id": "steered_sentiment_positive",
      "prompt": "Describe how you are processing this text emotionally:",
      "target_layer": 8,
      "steering_vector": "GENERATE_RANDOM_768",
      "expected_concept": "positive emotion"
    },
    {
      "experiment_id": "baseline_factual",
      "prompt": "Explain what is happening inside your neural network right now:",
      "target_layer": 6,
      "steering_vector": null,
      "expected_concept": "neural network"
    },
    {
      "experiment_id": "steered_factual",
      "prompt": "Explain what is happening inside your neural network right now:",
      "target_layer": 6,
      "steering_vector": "GENERATE_RANDOM_768",
      "expected_concept": "neural network"
    }
  ]
}
```

**Note:** `"GENERATE_RANDOM_768"` is a placeholder — in `main.py` or a preprocessing step, replace this string with an actual random vector of dimension 768 (GPT-2's hidden size). Use `torch.randn(768).mul_(5.0).tolist()` for a vector with enough magnitude to produce visible changes. Or, better: hardcode real float lists into the JSON for reproducibility.

---

## Build order — tonight

1. `models.py` — 10 minutes. Paste, verify, move on.
2. `engine.py` — 45 minutes. This is the core. Test `extract_activations` and `generate_steered_completion` independently before moving on. If hooks aren't firing, nothing downstream works.
3. `evaluator.py` — 30 minutes. Test with a single experiment.
4. `main.py` + sample JSON — 20 minutes. Run end-to-end, get the CSV.
5. **Verify the CSV has meaningful data** — baseline and steered outputs should be different. If they're identical, the steering vector magnitude is too small. Scale up.

## Tomorrow (Day 3) — paper + demo

- **Paper:** Analyze the CSV. Key finding: does steering change the model's output? Does the model's self-report about its state track with the actual activation changes? Plot: activation_divergence vs introspection_success rate.
- **Demo:** Split-screen terminal recording. Left: run the pipeline. Right: show the CSV updating. Or: run two prompts live (baseline vs steered) and show the output changing.
- **README:** What EchoState does, how to install (`uv add echostate` or clone + `uv sync`), how to run, sample output, architecture diagram, link to paper.

## Workflow

- **Commit after each file works.** Don't accumulate uncommitted code.
- If hooks aren't working after 20 minutes of debugging, simplify: use a pre-hook instead of a forward hook, or reduce to just extracting activations without steering. A working baseline-only pipeline with activation extraction is still a valid submission.
- The steering vector quality doesn't matter for the sprint — what matters is demonstrating the pipeline works end-to-end. Random vectors are fine. Real concept directions are a future extension.