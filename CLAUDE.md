# EchoState — working guide for Claude Code

## What this project is

EchoState tests whether a language model's self-report tracks its actual internal
state. It builds a **concept direction** contrastively, injects it into the
residual stream mid-forward with a PyTorch hook, and measures what changed:
internals, output text, and whether the model *said* it was in that state.

Built for the Apart Research Digital Minds Sprint (Tracks 3 and 4). The sprint is
over and submitted; the code is now a maintained library, not a race.

**The headline result:** across 6 models and 216 steered runs, concept directions
elicited concept vocabulary in 49/108 runs against 0/108 for magnitude-matched
random directions (Fisher exact p = 2.9e-18) — despite random directions moving
the internals *further*. This is a result about steering, **not** evidence of
introspection; see "Claims discipline" below.

## Repository layout

```
echostate/              the installable package
├── models.py           Pydantic schemas: IntrospectionExperiment, ArtifactResult, ExperimentSuite
├── engine.py           ActivationEngine ABC — the behaviour contract, torch-free
├── hook_engine.py      HookEngine: forward-hook implementation for any HF causal LM
├── steering.py         contrastive concept directions + magnitude-matched random controls
├── evaluator.py        orchestration and scoring; talks only to ActivationEngine
├── sweep.py            the multi-model experiment grid
├── analysis.py         pandas summaries over a sweep CSV
├── rendering.py        shared HTML/SVG rendering: templates, LineChart, sanitising
├── report.py           general-reader HTML report
├── paper.py            research-paper HTML + PDF (--pdf)
├── main.py             CLI: experiment JSON in, results CSV out
└── templates/          *.body.html markup, *.css styling, print.css for PDF
tests/                  34 tests; most run against a fake engine in milliseconds
experiments/            sample_experiments.json with literal fixed-seed vectors
notebooks/              executed analysis notebook
assets/                 project diagram (SVG) and PNG card
output/                 generated CSVs, HTML, PDF — gitignored except the submitted PDF/CSV
demo.py                 paced live demo for screen recording
VIDEO.md                video recording plan
```

## Commands

```bash
uv sync                                   # install everything
uv run pytest -q                          # 34 tests
uv run ruff check echostate tests demo.py # lint
uv run ruff format echostate tests demo.py

uv run python -m echostate.main experiments/sample_experiments.json output/results.csv
uv run python -m echostate.sweep output/sweep.csv
uv run python -m echostate.report output/sweep.csv output/report.html
uv run python -m echostate.paper output/sweep.csv output/paper.html --pdf output/paper.pdf
uv run python demo.py                     # live demo; --fast skips pauses
```

Console entry points: `echostate`, `echostate-sweep`, `echostate-report`.

## Architecture: the one boundary that matters

`evaluator.py` talks **only** to the `ActivationEngine` interface, never to torch.
That is why 27 of 34 tests run against a `FakeEngine` with no model download, and
why supporting a new model family is a subclass rather than a rewrite.

Do not import torch above that line. Do not let model-specific details leak into
`evaluator.py`, `sweep.py`, or `analysis.py`.

## Invariants — break these and you get wrong data, not errors

These are the failure modes that produced silent corruption during development.
Every one is covered by a test; keep it that way.

1. **A transformer block's output shape is version-dependent.** transformers 4.x
   returns a tuple `(hidden_states, kv_cache, ...)`; 5.x returns a bare Tensor.
   Always go through `_hidden_states()` / `_replace_hidden_states()` in
   `hook_engine.py`. Indexing `output[0]` directly under 5.x silently selects the
   batch dimension and returns a wrongly shaped tensor with no exception.

2. **Every registered hook must be removed in a `finally`.** A hook that survives
   an exception stays attached, and the next "clean baseline" run is silently
   steered. Covered by `test_hooks_do_not_leak_between_runs`.

3. **Steering vectors must match the model's dtype.** `pythia-410m` loads in
   float16; adding a float32 vector promotes the hidden states and the next layer
   norm dies on mixed dtypes. `_make_steering_hook` casts with `sv.to(hidden.dtype)`.

4. **Steering strength is relative, never absolute.** Residual norms differ by an
   order of magnitude across models. `build_concept_direction` returns a unit
   direction plus the layer's activation scale; `scaled()` combines them. A
   unit-length vector alone is roughly 20x too weak to do anything.

5. **Divergence must be read downstream of the injection.** Reading at
   `target_layer` returns exactly `baseline + steering_vector` and measures
   nothing. The evaluator defaults `read_layer` to the last block; the engine
   defaults it to `target_layer`. **These two defaults differ deliberately** —
   numbers from a direct engine call are not comparable with sweep numbers unless
   `read_layer` is passed explicitly.

6. **Scoring excludes words present in the prompt.** Without that, a model
   continuing "...inside your neural network:" by repeating "neural network"
   scores as a successful self-report. That confound inflated every control row in
   the first version. Covered by `test_words_echoed_from_the_prompt_do_not_count`.

7. **Decoding must stay greedy** (`do_sample=False`). With sampling, baseline and
   steered outputs differ every run regardless of the intervention, and the
   zero-vector and hook-leak tests become meaningless. Repetition loops are the
   cost of determinism and are handled with `no_repeat_ngram_size`, not sampling.

8. **`hidden_size` and `num_layers` come from the loaded model**, not constants.
   Validate layer index and vector width before inference — a wrong-width vector
   broadcasts silently instead of raising.

## Rendering

`render()` emits a **fragment** (title, style, body) for the Artifact host, which
supplies its own document shell. `render_standalone()` emits a **complete
document** plus `print.css`, for weasyprint PDF rendering.

Two print-only gotchas: weasyprint does not cascade a stylesheet into inline SVG
(so `LineChart` sets `font-size` as a presentation attribute), and it does not lay
out the legend as flex (so swatches need intrinsic dimensions in `print.css`).

Templates are package data, declared in `pyproject.toml`. Markup and CSS live in
`echostate/templates/`, never inside Python string literals.

## Conventions

- **uv**, never pip, for local work.
- Ruff, line length 92. `B008` and `SIM905` are ignored deliberately.
- Notebooks and Markdown are excluded from ruff: exploratory imports and prose
  are not lint targets.
- Docstrings explain *why*, especially the non-obvious failure mode a line
  prevents. Do not narrate what the code plainly does.
- Test each module as it is written. Prefer fake-engine tests; reserve real-model
  integration tests for `test_hook_engine.py`, which skips cleanly via
  `pytest.importorskip` when torch is absent.
- Verify generated output by rendering and looking at it, not by assuming.

## Claims discipline

The steering result is real and controlled. **It is not evidence of
introspection**, and no document in this repo should say otherwise. Steering
toward a concept raises the probability of that concept's tokens directly, so an
elicited report is confounded with the intervention's mechanical effect on the
output distribution.

A genuine introspection test needs the report and the injected concept to be
separable — forced choice over options supplied in the prompt, cross-concept
discrimination, or beating an external judge that sees only the model's output.
The README's final section covers this. Keep the caveat prominent in the README,
the paper, and the notebook.

## Packaging notes

- Import name and distribution name are both `echostate`.
- **`echostate` is already taken on PyPI by an unrelated project.** This package
  is not published; installing is `pip install git+https://github.com/guynutman/echo-state.git`.
  Do not add `pip install echostate` to any document.
- "Echo state network" is also an established reservoir-computing architecture.
  The README notes both collisions.
- `output/` is gitignored; the submitted PDF and sweep CSV were force-added.
