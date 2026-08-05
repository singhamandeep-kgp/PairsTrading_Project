import pandas as pd
import datetime as dt
import logging as log

# Import the MODULE, not the constant. `from ... import TRADING_DATES` would
# trigger the legacy pickle load at import time (via PEP 562 __getattr__), which
# is what made this module unimportable. Attribute access below defers it to the
# first actual call.
from statarb.backtest import calendar_legacy


def time_travel(date, lookback):
    # KNOWN DEFECT (behaviour unchanged here; recorded so it is not a surprise):
    # `trading_idx[i - lookback]` uses a negative index when lookback > i, and
    # numpy/pandas WRAP rather than raise -- so a large lookback silently returns
    # a date from the END of the calendar, i.e. from the future. The `except
    # IndexError` below never fires for that case.
    # DataAPI.trading_days_before() clamps instead, and is the correct version.
    trading_idx = pd.DatetimeIndex(calendar_legacy.TRADING_DATES)
    date = pd.to_datetime(date)
    if date in trading_idx:
        try:
            i = trading_idx.get_loc(date)
            return trading_idx[i - lookback].date()
        except IndexError:  # Handle invalid index due to lookback
            log.error(f"Lookback {lookback} pushes date {date} behind earliest available date.")
            raise
    else:
        log.error(f"Date {date} not found in TRADING_DATES.")
        raise ValueError(f"Date {date} is not in the trading dates calendar")