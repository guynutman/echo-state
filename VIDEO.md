# Video plan — 3 minutes

Recording checklist. Read the beats, don't read them word-for-word.

## Before you hit record (2 min)

```bash
# 1. Pre-warm the models so nothing downloads on camera
uv run python -c "from echostate import HookEngine; HookEngine('gpt2')"

# 2. Open these in two tabs
#    - the report page (artifact URL)
#    - the paper page (artifact URL)

# 3. Clear the terminal, make the font big (Ctrl+Shift+= a few times)
clear
```

Record the terminal segment **separately** and speed it up 3× in editing. Don't sit
through inference on camera.

---

## 0:00–0:20 — The question

> "Language models will tell you how they're feeling. I wanted to know whether
> those statements have anything to do with what's actually happening inside them.
> You can test that: change the internal state directly, and see if the self-report
> changes to match."

Screen: the README diagram, or just the title of the report page.

---

## 0:20–1:00 — Show it running

```bash
uv run python -m echostate.sweep output/demo.csv --models gpt2 --strengths 1.0
```

> "This is a forward hook reaching into GPT-2 at layer six and adding a vector to
> the residual stream while it generates. Every row is one steered run against its
> own baseline."

Let the progress lines scroll. Speed this up in the edit.

---

## 1:00–2:00 — The result *(the most important 60 seconds)*

Hold this table on screen. Don't scroll.

| arm | strength | activation div. | concept-vocab rate |
|---|---|---|---|
| **concept** | 0.25 → 1.0 | 0.207 → 0.480 | **33% → 61%** |
| random | 0.25 → 1.0 | 0.209 → 0.506 | **0% → 0%** |

> "Six models, 216 steered runs. Concept directions produce concept vocabulary 45%
> of the time. Random directions of *identical magnitude* — zero. Not once in 108
> runs. And look at the middle column: the random push moves the internals
> *further*. So this isn't about how hard you push. It's about direction."

Then show the actual completions:

> "Steering toward positive affect: 'gift', 'enjoy', smiley faces. The
> magnitude-matched random control at the same strength: noise."

---

## 2:00–2:40 — Why this isn't introspection *(your differentiator)*

> "Here's what I want to be careful about. This looks like introspection, and it
> isn't. Steering toward positive affect raises the probability of positive words
> directly — so the model saying 'enjoy' is the intervention surfacing
> mechanically, not the model noticing its own state and reporting it. My metric
> can't tell those apart.
>
> A real test needs the report and the injected concept to be separable — for
> instance, forced choice among options supplied in the prompt, where injected
> vocabulary can't inflate the answer."

---

## 2:40–3:00 — Close

> "So what I'm submitting is tooling, with its controls worked out and its
> confounds documented: contrastive directions, magnitude-matched controls, six
> models, three architecture families, and a harness that makes the real
> introspection experiment a config change rather than a rewrite."

---

## Rules

- **Lead with the caveat, don't bury it.** Calibration is the rarest thing in a
  sprint submission and the easiest way to stand out.
- **No slides.** A recording of the thing running proves the code works. Hold two
  static frames (the results table, the caveat) — that's it.
- Phone voice memo > laptop mic. Record audio separately if you can.
- One practice run, then record. Slightly rough natural speech beats a read script.
