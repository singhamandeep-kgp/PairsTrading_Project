import pandas as pd
import datetime as dt
from pairs_trading_pipeline.config.settings import TRADING_DATES
import logging as log


def time_travel(date, lookback):
    trading_idx = pd.DatetimeIndex(TRADING_DATES)
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