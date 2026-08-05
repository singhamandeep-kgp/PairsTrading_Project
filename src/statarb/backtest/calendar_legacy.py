"""
Legacy trading calendar, derived from a per-sector price pickle.

WHAT CHANGED AND WHY
--------------------
This module previously did its work at IMPORT time:

    read_path = os.path.join(os.getcwd(), 'GICS_Filtered_Equities_Prices')
    df = pd.read_pickle(os.path.join(read_path, "GICS_45.pkl"))
    TRADING_DATES = df.index.date.tolist()          # <- ran on import

Three separate problems with that, and together they made the entire legacy
strategy stack unimportable:

* it read a file that does not exist in this repository, so every import raised
  FileNotFoundError -- including imports that never wanted the calendar;
* it resolved the path from `os.getcwd()`, so behaviour depended on the directory
  the process happened to start in;
* it made the module impossible to import in a test, in CI, or on any machine
  without that pickle.

The data and the derivation are unchanged. The only difference is that they now
happen when someone ASKS for them, via `trading_dates()` / `rebalance_dates()` /
`selection_dates()`. Results are cached, so repeated calls cost nothing.

PREFER THE REAL CALENDAR
------------------------
For new code use `DataAPI.get_trading_days()` and
`DataAPI.get_rebalance_dates()`. Those come from FactSet's own exchange calendar
(`ref_calendar_dates` + `ref_calendar_holidays`) rather than from one stock's
observed price dates -- which, being one stock's dates, also encode that stock's
trading halts as though they were market holidays.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

DEFAULT_PICKLE_DIR = "GICS_Filtered_Equities_Prices"
DEFAULT_PICKLE_NAME = "GICS_45.pkl"


def _panel_path(pickle_dir: str | Path | None = None,
                pickle_name: str = DEFAULT_PICKLE_NAME) -> Path:
    base = Path(pickle_dir) if pickle_dir is not None else Path(os.getcwd()) / DEFAULT_PICKLE_DIR
    return base / pickle_name


@lru_cache(maxsize=4)
def _panel(pickle_dir: str | None = None,
           pickle_name: str = DEFAULT_PICKLE_NAME) -> pd.DataFrame:
    """Load the sector price panel. Raises a clear error if it is absent."""
    path = _panel_path(pickle_dir, pickle_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Legacy calendar panel not found: {path}\n"
            "This module needs a GICS_*.pkl produced by the old pickle pipeline. "
            "For new code use DataAPI.get_trading_days() / get_rebalance_dates(), "
            "which read FactSet's own exchange calendar and need no pickle."
        )
    return pd.read_pickle(path)


@lru_cache(maxsize=4)
def trading_dates(pickle_dir: str | None = None) -> list:
    """Trading dates, taken from the panel's index."""
    return _panel(pickle_dir).index.date.tolist()


@lru_cache(maxsize=4)
def rebalance_dates(pickle_dir: str | None = None) -> list:
    """First trading day of each month."""
    df = _panel(pickle_dir)
    first = df.groupby(df.index.to_period("M")).apply(lambda x: x.index.min())
    return first.dt.date.tolist()


@lru_cache(maxsize=4)
def selection_dates(pickle_dir: str | None = None) -> list:
    """One trading day before each rebalance date."""
    idx = pd.DatetimeIndex(trading_dates(pickle_dir))
    out = []
    for d in rebalance_dates(pickle_dir)[1:]:
        d = pd.to_datetime(d)
        if d in idx:
            out.append(idx[idx.get_loc(d) - 1].date())
    return out


def __getattr__(name: str):
    """Module-level attribute access, so `from ... import TRADING_DATES` still works.

    PEP 562 lets the old constant names keep working for existing callers while
    the underlying load stays lazy: the pickle is only touched if someone actually
    reads one of these names.
    """
    if name == "TRADING_DATES":
        return trading_dates()
    if name == "REBALANCE_DATES":
        return rebalance_dates()
    if name == "SELECTION_DATES":
        return selection_dates()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
