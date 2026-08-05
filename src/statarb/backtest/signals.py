import pandas as pd
import numpy as np


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def generate_positions(
    spread: pd.Series,
    z_window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float | None = None,
    max_holding_days: int | None = None,
) -> pd.Series:
    z = rolling_zscore(spread, z_window)
    pos = []
    state = 0
    hold = 0
    for t, zt in z.items():
        if np.isnan(zt):
            pos.append(0)
            state = 0
            hold = 0
            continue
        if state == 0:
            if zt > entry_z:
                state = -1
                hold = 1
            elif zt < -entry_z:
                state = 1
                hold = 1
        elif state == 1:  # long spread
            exit_cond = zt > -exit_z or (stop_z is not None and zt < -abs(stop_z))
            if exit_cond or (max_holding_days and hold >= max_holding_days):
                state = 0
                hold = 0
            else:
                hold += 1
        elif state == -1:  # short spread
            exit_cond = zt < exit_z or (stop_z is not None and zt > abs(stop_z))
            if exit_cond or (max_holding_days and hold >= max_holding_days):
                state = 0
                hold = 0
            else:
                hold += 1
        pos.append(state)
    return pd.Series(pos, index=spread.index)
