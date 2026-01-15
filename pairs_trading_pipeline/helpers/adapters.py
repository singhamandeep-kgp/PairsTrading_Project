import os
from typing import Tuple, Optional

import pandas as pd

from pairs_trading_pipeline.data.data_processor import DataProcessor
from pairs_trading_pipeline.features.pca import apply_pca as _apply_pca
from pairs_trading_pipeline.features.clustering import optics_cluster as _optics_cluster
from pairs_trading_pipeline.models.cointegration import _align_prices as _align_prices
from pairs_trading_pipeline.models.cointegration import _load_log_price_series as _load_log_price_series


def load_sector_prices(sector_id: int, prices_dir: str = "GICS_Filtered_Equities_Prices") -> pd.DataFrame:
    file_path = os.path.join(os.getcwd(), prices_dir, f"GICS_{sector_id}.pkl")
    return pd.read_pickle(file_path)


def scale_returns(prices: pd.DataFrame) -> pd.DataFrame:
    processor = DataProcessor(prices)
    return processor.scaler(prices)


def pca_on_returns(returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    factor_returns, factor_loadings, evr = _apply_pca(returns)
    return factor_returns, factor_loadings, pd.Series(evr)


def optics_cluster(loadings: pd.DataFrame, **kwargs) -> pd.Series:
    return _optics_cluster(loadings, **kwargs)


def align_prices(p1: pd.Series, p2: pd.Series, min_obs: int = 50) -> Optional[pd.DataFrame]:
    return _align_prices(p1, p2, min_obs=min_obs)


def load_log_price_series(permno: int, prices_dir: str = "GICS_Filtered_Equities_Prices") -> pd.Series:
    return _load_log_price_series(permno, prices_dir=prices_dir)
