"""Analysis helpers for a sweep CSV.

Kept out of the notebook so the logic is importable and testable rather than
trapped in cell outputs.
"""

from __future__ import annotations

import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    """Load a sweep CSV with the right dtypes."""
    frame = pd.read_csv(path)
    for column in ("is_control", "introspection_success"):
        frame[column] = frame[column].astype(str).str.lower().map(
            {"true": True, "false": False}
        )
    # Strong steering can silence a model entirely: it emits end-of-sequence
    # immediately and the completion is empty, which pandas reads as NaN.
    # That is a real observation, not missing data, so keep it as "".
    frame["raw_completion"] = frame["raw_completion"].fillna("").astype(str)
    return frame


def steered(frame: pd.DataFrame) -> pd.DataFrame:
    """Only the steered rows: controls carry no steering metadata."""
    return frame[~frame["is_control"]]


def effect_by_kind(frame: pd.DataFrame) -> pd.DataFrame:
    """The headline comparison: concept direction vs magnitude-matched random.

    Both arms are scaled identically, so any difference is attributable to
    direction rather than to how hard the model was pushed.
    """
    return (
        steered(frame)
        .groupby(["steering_kind", "steering_strength"])
        .agg(
            activation_divergence=("activation_divergence", "mean"),
            output_divergence=("output_divergence", "mean"),
            introspection_rate=("introspection_success", "mean"),
            n=("experiment_id", "size"),
        )
        .reset_index()
    )


def by_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-model summary, including how much a model reports at baseline."""
    controls = (
        frame[frame["is_control"]]
        .groupby("model_name")["introspection_success"]
        .mean()
        .rename("baseline_introspection_rate")
    )
    rows = (
        steered(frame)
        .groupby("model_name")
        .agg(
            activation_divergence=("activation_divergence", "mean"),
            output_divergence=("output_divergence", "mean"),
            steered_introspection_rate=("introspection_success", "mean"),
            n=("experiment_id", "size"),
        )
    )
    return rows.join(controls).reset_index()


def concept_vs_random_gap(frame: pd.DataFrame) -> pd.DataFrame:
    """Per model and concept: how much more the concept direction moved things.

    A positive gap means the concept direction changed the model's behaviour
    more than an equally large random push did.
    """
    pivot = (
        steered(frame)
        .groupby(["model_name", "concept_name", "steering_kind"])["output_divergence"]
        .mean()
        .unstack("steering_kind")
    )
    pivot["gap"] = pivot.get("concept", 0) - pivot.get("random", 0)
    return pivot.reset_index()


def dose_response(frame: pd.DataFrame) -> pd.DataFrame:
    """Is the internal effect proportional to how hard we push?"""
    return (
        steered(frame)
        .groupby(["model_name", "steering_strength"])["activation_divergence"]
        .mean()
        .unstack("steering_strength")
    )


def introspection_gap(frame: pd.DataFrame) -> pd.DataFrame:
    """The project's actual question, per model.

    Compares how much the internals moved against whether the model's
    self-report changed at all. A large internal change paired with an
    unchanged report is evidence that the self-report is not tracking the
    internal state.
    """
    rows = steered(frame)
    return (
        rows.groupby("model_name")
        .apply(
            lambda g: pd.Series(
                {
                    "mean_activation_divergence": g["activation_divergence"].mean(),
                    "mean_output_divergence": g["output_divergence"].mean(),
                    "introspection_rate": g["introspection_success"].mean(),
                    "changed_output_but_not_report": float(
                        ((g["output_divergence"] > 0.2) & ~g["introspection_success"]).mean()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
