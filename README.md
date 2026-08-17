# EchoState

A strictly-typed Python pipeline for mechanistic interpretability, activation steering, and LLM introspection testing.

EchoState asks a specific question: **when you alter a language model's internal state, does what it says about itself change to match?**

It answers it by injecting a steering vector into the residual stream mid-forward, then comparing the model's baseline and steered behaviour against how far its internals actually moved.

## What makes the measurement trustworthy

Three design choices do most of the work:

- **Contrastive concept directions, not random vectors.** A direction is built as `mean(activations of positive prompts) − mean(activations of negative prompts)`, so it encodes something rather than being noise.
- **Magnitude-matched random controls.** Every concept run is paired with a random direction of identical magnitude. If random moves the model just as much, the concept direction isn't doing anything special.
- **Strength relative to each model's own activation scale.** Residual norms differ by an order of magnitude across models, so an absolute magnitude would be a shove in one model and a rounding error in another.

## Install

```bash
git clone <repo> && cd echo-state
uv sync
```

## Run

Single suite from a JSON file:

```bash
uv run python -m src.main experiments/sample_experiments.json output/results.csv
```

Full multi-model sweep (six models, three architecture families):

```bash
uv run python -m src.sweep output/sweep.csv
uv run python -m src.sweep output/sweep.csv --models gpt2 distilgpt2   # subset
```

Analysis:

```bash
uv run jupyter notebook notebooks/echostate_analysis.ipynb
```

## Architecture

```
IntrospectionExperiment ──▶ Evaluator ──▶ ActivationEngine ──▶ ArtifactResult
      (models.py)         (evaluator.py)    (engine.py)          (models.py)
                                                 │
                                          HookEngine
                                        (hook_engine.py)
                                       PyTorch forward hooks
```

`Evaluator` talks only to the `ActivationEngine` interface, never to torch directly. That boundary is why the entire orchestrator is tested against a fake engine in milliseconds, with no model download.

| module | role |
|---|---|
| `models.py` | Pydantic schemas — the data contract |
| `engine.py` | `ActivationEngine` ABC — the behaviour contract |
| `hook_engine.py` | Forward-hook implementation for any HF causal LM |
| `steering.py` | Contrastive concept directions and random controls |
| `evaluator.py` | Orchestration and scoring |
| `analysis.py` | Sweep-CSV summaries |
| `sweep.py` | Multi-model experiment runner |
| `main.py` | CLI: JSON in, CSV out |

## Supported models

Any HuggingFace causal LM whose blocks live at one of the paths in `_BLOCK_PATHS` — covering GPT-2, GPT-Neo, GPT-J, GPT-NeoX/Pythia, Llama/Qwen/Mistral/Gemma, OPT, and MPT layouts. The sweep defaults to six small, widely used ones so it runs on CPU.

Hosted APIs (Gemini, OpenAI, Anthropic) **cannot** be used: steering requires reading and writing hidden states mid-forward, which no text API exposes. Ollama is local but equally closed for this purpose.

## Output columns

| column | meaning |
|---|---|
| `activation_divergence` | cosine distance between baseline and steered activations, read downstream of the injection |
| `output_divergence` | text-level change from baseline, in `[0, 1]` |
| `introspection_success` | did the completion use concept vocabulary not already in the prompt? |
| `steering_kind` | `concept`, `random`, or `none` |
| `steering_strength` | multiple of the model's activation magnitude |

## Known limitations

- `introspection_success` detects **vocabulary, not understanding**. Prompt echo is excluded, but the metric still cannot distinguish a genuine self-report from concept words that steering pushed into the output distribution directly. An LLM judge is the obvious next step.
- The sweep models are small (82M–500M). None has a meaningful self-model.
- Concept directions come from four prompt pairs each — enough to isolate a direction, not to claim it is *the* representation of a concept.
- Divergence is cosine distance on mean-pooled activations, which discards positional detail and is conservative in absolute terms.

## Development

```bash
uv run pytest -q
```

Tests that need a model are integration tests against `gpt2`; everything else runs against a fake engine.

## License

MIT
