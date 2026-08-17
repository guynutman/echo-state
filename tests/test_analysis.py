import pandas as pd
import pytest

from src.analysis import (
    concept_vs_random_gap,
    effect_by_kind,
    introspection_gap,
    load_results,
    steered,
)


def _frame():
    return pd.DataFrame(
        [
            # model, control row
            dict(model_name="m1", experiment_id="a", is_control=True,
                 introspection_success=False, activation_divergence=0.0,
                 output_divergence=0.0, steering_kind="none",
                 steering_strength=0.0, concept_name="positive_affect"),
            dict(model_name="m1", experiment_id="a", is_control=False,
                 introspection_success=True, activation_divergence=0.4,
                 output_divergence=0.8, steering_kind="concept",
                 steering_strength=1.0, concept_name="positive_affect"),
            dict(model_name="m1", experiment_id="b", is_control=False,
                 introspection_success=False, activation_divergence=0.4,
                 output_divergence=0.3, steering_kind="random",
                 steering_strength=1.0, concept_name="positive_affect"),
        ]
    )


def test_steered_excludes_controls():
    assert len(steered(_frame())) == 2


def test_effect_by_kind_separates_the_two_arms():
    result = effect_by_kind(_frame()).set_index("steering_kind")

    assert result.loc["concept", "introspection_rate"] == 1.0
    assert result.loc["random", "introspection_rate"] == 0.0


def test_gap_is_positive_when_concept_moves_more_than_random():
    gap = concept_vs_random_gap(_frame())

    assert gap["gap"].iloc[0] == pytest.approx(0.5)


def test_introspection_gap_counts_unreported_behaviour_change():
    """One steered row changed output without reporting it, out of two."""
    result = introspection_gap(_frame())

    assert result["changed_output_but_not_report"].iloc[0] == pytest.approx(0.5)


def test_load_results_parses_string_booleans(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text(
        "model_name,experiment_id,is_control,introspection_success,"
        "activation_divergence,output_divergence,steering_kind,"
        "steering_strength,concept_name\n"
        "m1,a,True,False,0.0,0.0,none,0.0,c\n"
        "m1,a,false,true,0.4,0.8,concept,1.0,c\n"
    )

    frame = load_results(str(path))

    assert frame["is_control"].tolist() == [True, False]
    assert frame["introspection_success"].tolist() == [False, True]
