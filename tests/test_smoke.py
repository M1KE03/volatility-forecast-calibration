"""Structural smoke tests for the scaffold.

These assert only that the repository is laid out as designed and that every module
imports. They deliberately do **not** test behaviour: each module's behaviour is tested
in its own file, and duplicating any of it here would mean two places to update and one
of them going stale.

Written at Stage 1 against a scaffold in which every function body raised
``NotImplementedError``. Both categories it promised have since arrived and live where
they belong -- transformation tests in ``test_data.py``, ``test_models.py`` and
``test_bootstrap.py``, look-ahead tests in ``test_data.py`` and ``test_backtest.py``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "src",
    "src.data",
    "src.models",
    "src.backtest",
    "src.evaluation",
    "src.bootstrap",
]

EXPECTED_DIRS = [
    "data/raw",
    "data/processed",
    "src",
    "tests",
    "notebooks",
    "figures",
    "report",
    "docs",
]

EXPECTED_FILES = [
    "README.md",
    "requirements.txt",
    "run_all.py",
    "research_log.md",
    "report/report.md",
    "docs/project1-implementation-plan.md",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    """Every source module imports cleanly."""
    assert importlib.import_module(module_name) is not None


@pytest.mark.parametrize("rel_path", EXPECTED_DIRS)
def test_expected_directory_exists(rel_path: str) -> None:
    """The repository layout matches the design."""
    assert (PROJECT_ROOT / rel_path).is_dir()


@pytest.mark.parametrize("rel_path", EXPECTED_FILES)
def test_expected_file_exists(rel_path: str) -> None:
    """Top-level project files are present."""
    assert (PROJECT_ROOT / rel_path).is_file()


def test_entry_point_parser_builds() -> None:
    """run_all.py exposes every stage through its parser."""
    run_all = importlib.import_module("run_all")
    parser = run_all.build_parser()
    args = parser.parse_args(["--stage", "data"])
    assert args.stage == "data"
    assert set(run_all.STAGES) == set(run_all.STAGE_ORDER)


# There was a `test_unimplemented_stages_still_raise` here, asserting that stages not yet
# built failed loudly rather than appearing to succeed. Its list shrank by one entry per
# stage -- `data` and `eda` at Stage 0, `backtest` at Stage 1, `bayes` at Stage 3 -- and
# emptied at Stage 4 when `evaluate` and `figures` landed. It is removed rather than left
# parametrised over nothing, which collects zero tests and quietly asserts nothing.
# Behaviour for an implemented stage is tested in that stage's own test module.
