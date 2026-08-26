"""Test configuration: the opt-in gate for the real-sampler look-ahead audit.

``test_bayesian_look_ahead_with_the_real_sampler`` runs the audit with NUTS actually
sampling and takes about forty minutes. Everything it covers beyond the default suite is
the sampler's own internals -- the Bayesian track's look-ahead surfaces are audited on
every run against a recording stub (decision D26) -- so it is opt-in.

**A marker alone would not be enough, and the reason is a trap worth recording.** With
the deselection expressed as ``addopts = -m "not bayes_audit"`` in ``pytest.ini``, any
command line carrying its own ``-m`` replaces it rather than adding to it: ``pytest -m
"not slow"``, the documented fast inner loop, would *select* the forty-minute test. It
did, once, before this file existed. A gate that a plausible everyday command silently
disarms is not a gate, so the decision is moved to an explicit flag that no ``-m`` can
touch:

    pytest --bayes-audit -m bayes_audit     # the audit alone, ~40 minutes
    pytest --bayes-audit                    # everything, including it
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--bayes-audit",
        action="store_true",
        default=False,
        help=(
            "Run the look-ahead audit with the real NUTS sampler (~40 minutes). "
            "Skipped without this flag regardless of any -m selection."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--bayes-audit"):
        return
    skip = pytest.mark.skip(
        reason="needs --bayes-audit; ~40 minutes with NUTS sampling. The Bayesian "
        "track's look-ahead surfaces are audited on every run against a recording "
        "stub (D26); this adds the end-to-end confirmation."
    )
    for item in items:
        if "bayes_audit" in item.keywords:
            item.add_marker(skip)
