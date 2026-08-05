import pandas as pd
import numpy as np


def pair_pnl(spread: pd.Series, positions: pd.Series) -> pd.Series:
    spread_ret = spread.diff().fillna(0.0)
    pos_lag = positions.shift(1).fillna(0.0)
    return pos_lag * spread_ret


def portfolio_pnl(pair_pnl_df: pd.DataFrame) -> pd.Series:
    if pair_pnl_df.empty:
        return pd.Series(dtype=float)
    return pair_pnl_df.mean(axis=1)


def equity_curve(portfolio_pnl_series: pd.Series, start_equity: float = 1.0) -> pd.Series:
    return start_equity + portfolio_pnl_series.cumsum()
