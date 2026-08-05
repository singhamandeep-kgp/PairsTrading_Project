"""
Shared pytest fixtures.

The warehouse is built ONCE per session into a temporary directory and shared.
Two things follow from that, both deliberate:

* `DataAPI(root=...)` takes the root explicitly, so no test ever mutates
  `FACTSET_DEST_ROOT` or monkeypatches a module before importing it. Those
  ordering games are how a test ends up passing for the wrong reason.

* Nothing touches the real ~/statarb-data. The suite runs identically on a
  machine that has never had FactSet access, which is the whole point.
"""

from __future__ import annotations

import pytest

from tests.fixtures.synthetic import build_synthetic_warehouse


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory):
    """A complete synthetic Delta warehouse. Built once, reused by every test."""
    root = tmp_path_factory.mktemp("statarb-data")
    loc, planted = build_synthetic_warehouse(root)
    return loc, planted


@pytest.fixture(scope="session")
def planted(warehouse):
    """The sids carrying each planted property. See tests/fixtures/synthetic.py."""
    return warehouse[1]


@pytest.fixture(scope="session")
def loc(warehouse):
    return warehouse[0]


@pytest.fixture
def api(warehouse):
    """A DataAPI bound to the synthetic warehouse via an explicit root."""
    from statarb.data.api import DataAPI

    return DataAPI(root=warehouse[0])
