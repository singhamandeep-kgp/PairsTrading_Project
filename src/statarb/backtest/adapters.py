import os
from typing import Tuple, Optional

import pandas as pd

from statarb.research.scaling import DataProcessor
from statarb.research.pca import apply_pca as _apply_pca
from statarb.research.clustering import optics_cluster as _optics_cluster
from statarb.research.cointegration import _align_prices as _align_prices
from statarb.research.cointegration import load_log_prices_by_glob as _load_log_prices_by_glob


def load_sector_prices(sector_id: int, prices_dir: str = "GICS_Filtered_Equities_Prices") -> pd.DataFrame:
    file_path = os.path.join(os.getcwd(), prices_dir, f"GICS_{sector_id}.pkl")
    return pd.read_pickle(file_path)


def scale_returns(prices: pd.DataFrame) -> pd.DataFrame:
    processor = DataProcessor(prices)
    return processor.scaler(prices)


def pca_on_returns(returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    factor_returns, factor_loadings, evr = _apply_pca(returns)
    return factor_returns, factor_loadings, pd.Series(evr)


def optics_cluster_adapter(loadings: pd.DataFrame, **kwargs) -> pd.Series:
    return _optics_cluster(loadings, **kwargs)


def align_prices(p1: pd.Series, p2: pd.Series, min_obs: int = 50) -> Optional[pd.DataFrame]:
    return _align_prices(p1, p2, min_obs=min_obs)


def load_log_prices_by_glob(permno: int, prices_dir: str = "GICS_Filtered_Equities_Prices") -> pd.Series:
    return _load_log_prices_by_glob(permno, prices_dir=prices_dir)
