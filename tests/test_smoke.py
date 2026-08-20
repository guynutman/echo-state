"""Every module imports, and its public surface is present.

The rest of the suite exercises `models`, `engine`, `evaluator`, `analysis` and
`main`. Nothing imports `sweep`, `report`, `paper`, `rendering` or `steering`,
so an import-level break in those — a bad name after a refactor, a circular
import — would otherwise pass CI silently.

Modules are grouped by what they actually need, so this file still runs when the
optional analysis extra is absent.
"""

import importlib

import pytest

CORE = ["models", "engine", "evaluator", "rendering", "main"]
NEEDS_TORCH = ["hook_engine", "steering", "sweep"]
NEEDS_PANDAS = ["analysis", "report", "paper"]


def _import(name):
    return importlib.import_module(f"echostate.{name}")


@pytest.mark.parametrize("name", CORE)
def test_core_modules_import(name):
    assert _import(name) is not None


@pytest.mark.parametrize("name", NEEDS_TORCH)
def test_torch_modules_import(name):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    assert _import(name) is not None


@pytest.mark.parametrize("name", NEEDS_PANDAS)
def test_analysis_modules_import(name):
    pytest.importorskip("pandas")
    assert _import(name) is not None


def test_package_exports_its_public_api():
    import echostate

    for name in echostate.__all__:
        assert hasattr(echostate, name), f"{name} is in __all__ but not importable"


def test_console_entry_points_are_callable():
    """Each script in pyproject.toml must resolve to a real callable."""
    pytest.importorskip("pandas")

    for module_name in ("main", "sweep", "report"):
        assert callable(_import(module_name).main)
