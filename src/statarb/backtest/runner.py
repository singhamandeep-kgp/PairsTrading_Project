import pandas as pd
# Module import, not `from ... import TRADING_DATES`: the latter triggers the
# legacy pickle load at import time. Attribute access defers it to first use.
from statarb.backtest import calendar_legacy
from statarb.backtest.portfolio import compute_portfolio
import datetime as dt

start_date = None
end_date = None
custom_first_selection = None
custom_first_rebalance = None
starting_strategy_value = 100
pca_lookback = 252
coint_lookback = 504
z_score_lookback = 60

def run_backtest(start_date = start_date, end_date = end_date,
                 custom_first_selection = custom_first_selection,
                 custom_first_rebalance = custom_first_rebalance,
                 starting_strategy_value = starting_strategy_value,
                 pca_lookback = pca_lookback,
                 coint_lookback = coint_lookback, 
                 z_score_lookback = z_score_lookback):

    for date in calendar_legacy.TRADING_DATES:
        if date in calendar_legacy.SELECTION_DATES:
            new_portfolio = compute_portfolio(date, pca_lookback, coint_lookback, z_score_lookback)
        if date in calendar_legacy.REBALANCE_DATES:
            portfolio = new_portfolio 
        
        #starting_strategy_value +=