import pandas as pd
from pairs_trading_pipeline.config.settings import TRADING_DATES, SELECTION_DATES, REBALANCE_DATES
from pairs_trading_pipeline.backtesting_engine.portfolio_builder import compute_portfolio
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

    for date in TRADING_DATES:
        if date in SELECTION_DATES:
            new_portfolio = compute_portfolio(date, pca_lookback, coint_lookback, z_score_lookback)
        if date in REBALANCE_DATES:
            portfolio = new_portfolio 
        
        #starting_strategy_value +=