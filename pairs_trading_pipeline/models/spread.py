import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from functools import lru_cache
from pairs_trading_pipeline.models.cointegration import _align_prices
import statsmodels.formula.api as sm


class ModelSpread():
    
    def __init__(self, candidates_path = 'GICS_Filtered_Pair_Candidates') -> None:

        self.candidates_path = candidates_path

    @lru_cache(maxsize=None)
    def _load_log_price_series(self, permno: int, prices_dir: str = "GICS_Filtered_Equities_Prices", sector = None) -> pd.Series:

        prices_dir = os.path.join(os.getcwd(), prices_dir)
        file_path = os.path.join(prices_dir, f"GICS_{sector}.pkl")
        df = pd.read_pickle(file_path)    
        series = df[permno].copy().sort_index().astype(float)
        return np.log(series)

    def get_all_pairs(self):
        
        data_dir = os.path.join(os.getcwd(), self.candidates_path)
        all_files = sorted(glob.glob(os.path.join(data_dir, 'GICS_*.csv')))
        cointegrated_pairs = None
        for file in all_files:
            df = pd.read_csv(file)
            df.drop(columns = "Unnamed: 0", inplace = True)
            path = file
            sector = int(Path(path).stem.split("_")[1])
            sectorwise_cointegrated_pairs = df[df['cointegrated'] == bool(True)].copy()
            sectorwise_cointegrated_pairs['GICS_Sector'] = sector
            if cointegrated_pairs is None:
                cointegrated_pairs = sectorwise_cointegrated_pairs 
            else:
                cointegrated_pairs = pd.concat([cointegrated_pairs, sectorwise_cointegrated_pairs],ignore_index=True)
        return cointegrated_pairs
    
    def compute_spread(self, prices_df, alpha, beta):

        prices_df["spread"] = prices_df["asset1"] - (alpha + beta * prices_df["asset2"])
        return prices_df
    
    def compute_spread_volatility(self, spread_df, df, index):

        spread = spread_df["spread"].dropna()
        sigma_level = float(spread.std())
        sigma_change = float(spread.diff().dropna().std())
        df.at[index, "spread_sigma"] = sigma_level * np.sqrt(252)
        df.at[index, "spread_change_sigma"] = sigma_change * np.sqrt(252)

    def compute_spread_half_life(self, spread_df, df, index):

        s = spread_df["spread"].dropna()

        df_reg = pd.DataFrame({"s_lag": s.shift(1),"ds": s.diff()}).dropna()

        res = sm.ols("ds ~ s_lag", data=df_reg).fit()
        phi = float(res.params["s_lag"])

        if phi >= 0:
            df.at[index, "spread_half_life"] = np.inf
            return

        df.at[index, "spread_half_life"] = float(-np.log(2) / phi)
        
    def hedge_ratio_ols(self, prices_df, df, index):
        
        formula = "asset1 ~ asset2"
        res = sm.ols(formula = formula, data=prices_df).fit()
        alpha = float(res.params["Intercept"])
        beta = float(res.params["asset2"])

        df.at[index, "hedge_ratio_OLS_alpha"] = alpha
        df.at[index, "hedge_ratio_OLS_beta"] = beta
        df.at[index, "hedge_ratio_OLS_r2"] = float(res.rsquared)
        df.at[index, "hedge_ratio_OLS_resid_std"] = float(res.resid.std()) 
        return alpha, beta

    def compute_spread_stats(self):

        df = self.get_all_pairs()
        df = df[["permno1", "permno2", "cointegrated", "p_value", "GICS_Sector"]]

        for index ,row in df.iterrows():
            a = row.permno1
            b = row.permno2
            sector = row.GICS_Sector

            s1 = self._load_log_price_series(permno = a, sector = sector)
            s2 = self._load_log_price_series(permno = b, sector = sector)
            aligned = _align_prices(s1, s2)

            alpha, beta = self.hedge_ratio_ols(aligned, df, index)
            aligned = self.compute_spread(aligned, alpha, beta)
            self.compute_spread_volatility(aligned, df, index)
            self.compute_spread_half_life(aligned, df, index)
        
        return df




        
