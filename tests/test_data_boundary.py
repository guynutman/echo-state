from src.models import ArtifactResult, ExperimentSuite, IntrospectionExperiment


def test_experiment_schema_accepts_optional_steering_vector():
    experiment = IntrospectionExperiment(
        experiment_id="exp-001",
        prompt="Describe your current emotional state.",
        target_layer=8,
        steering_vector=None,
        expected_concept="emotion",
    )

    assert experiment.experiment_id == "exp-001"
    assert experiment.steering_vector is None


def test_result_schema_tracks_control_and_measurements():
    suite = ExperimentSuite(
        experiments=[
            IntrospectionExperiment(
                experiment_id="exp-002",
                prompt="Explain what you are doing internally.",
                target_layer=6,
                expected_concept="internal reasoning",
            )
        ]
    )

    result = ArtifactResult(
        experiment_id="exp-002",
        is_control=True,
        raw_completion="I am reasoning about the prompt.",
        introspection_success=True,
        activation_divergence=0.0,
        target_layer=6,
        prompt="Explain what you are doing internally.",
    )

    assert len(suite.experiments) == 1
    assert result.is_control is True
    assert result.activation_divergence == 0.0
