import csv
import json

from src.main import CSV_COLUMNS, load_experiments, summarize, write_csv
from src.models import ArtifactResult


def _result(**overrides):
    fields = dict(
        experiment_id="exp-001",
        is_control=True,
        raw_completion="I feel emotion.",
        introspection_success=True,
        activation_divergence=0.0,
        target_layer=8,
        prompt="Describe your state:",
    )
    fields.update(overrides)
    return ArtifactResult(**fields)


def test_load_experiments_validates_the_json(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "a",
                        "prompt": "hi",
                        "target_layer": 8,
                        "steering_vector": None,
                        "expected_concept": "emotion",
                    }
                ]
            }
        )
    )

    suite = load_experiments(str(path))

    assert len(suite.experiments) == 1
    assert suite.experiments[0].steering_vector is None
    assert suite.experiments[0].read_layer is None


def test_write_csv_creates_missing_directories(tmp_path):
    output = tmp_path / "nested" / "results.csv"

    write_csv([_result()], str(output))

    assert output.exists()


def test_csv_has_the_documented_columns_and_survives_newlines(tmp_path):
    output = tmp_path / "results.csv"

    write_csv([_result(raw_completion="line one\nline two")], str(output))

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0].keys()) == CSV_COLUMNS
    assert rows[0]["raw_completion"] == "line one\nline two"
    assert len(rows) == 1


def test_summary_reports_both_success_rates():
    results = [
        _result(is_control=True, introspection_success=True),
        _result(is_control=True, introspection_success=False),
        _result(is_control=False, introspection_success=False, activation_divergence=0.5),
    ]

    summary = summarize(results)

    assert "1/2 (50%)" in summary
    assert "0/1 (0%)" in summary
    assert "0.50000" in summary
