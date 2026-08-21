"""Structural smoke tests for the scaffold.

These assert only that the repository is laid out as designed and that every module
imports. They deliberately do **not** test behaviour, because there is no behaviour
yet — every function body raises ``NotImplementedError``.

Real tests arrive with each implementation stage. Two categories are planned and are
recorded here so they are not forgotten:

- Transformation tests: returns, the Parkinson estimator, the EWMA recursion, the
  GARCH filter, and the stationary bootstrap's block-length distribution.
- Look-ahead tests: that a forecast for date t is unchanged when every observation
  from t onward is corrupted. That is the strongest available check that no future
  information leaks into a forecast, and it will be applied to the analysis frame, the
  estimation-window slicing, and the regime labels.
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
]

EXPECTED_FILES = [
    "README.md",
    "requirements.txt",
    "run_all.py",
    "research_log.md",
    "report/report.md",
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


def test_stages_are_stubs() -> None:
    """Every stage is still an unimplemented stub.

    This test is expected to be deleted, one stage at a time, as stages land. Its
    failure is a signal to update the scaffold's story, not a bug.
    """
    run_all = importlib.import_module("run_all")
    parser = run_all.build_parser()
    args = parser.parse_args(["--stage", "data"])
    for name, fn in run_all.STAGES.items():
        with pytest.raises(NotImplementedError):
            fn(args)
