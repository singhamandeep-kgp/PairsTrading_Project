import pandas as pd
import os

###########################################################

# Taking Apple's trading dates as trading calendar

# First_Date = 2019-01-02
# Last_Date = 2024-12-30

read_path = os.path.join(os.getcwd(), 'GICS_Filtered_Equities_Prices')
read_file = os.path.join(read_path, "GICS_45.pkl")
df = pd.read_pickle(read_file)
TRADING_DATES = df.index.date.tolist()

###########################################################

# Rebalancing happens on first trading day of the month 

rebalance_dates = (df.groupby(df.index.to_period("M")).apply(lambda x: x.index.min()))
REBALANCE_DATES = rebalance_dates.dt.date.tolist()

###########################################################

# Selection of new portfolio happens 1 day before rebalancing

trading_idx = pd.DatetimeIndex(TRADING_DATES)
SELECTION_DATES = []
for date in REBALANCE_DATES[1:]:
    date = pd.to_datetime(date)
    if date in trading_idx:
        i = trading_idx.get_loc(date) 
        SELECTION_DATES.append(trading_idx[i - 1].date())