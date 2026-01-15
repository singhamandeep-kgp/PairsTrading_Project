from pairs_trading_pipeline.data.dates_helper import time_travel
from pairs_trading_pipeline.features.pca import apply_pca, find_relevant_PCs
from pairs_trading_pipeline.features.clustering import optics_cluster
from pairs_trading_pipeline.config.settings import TRADING_DATES, SELECTION_DATES, REBALANCE_DATES
from pairs_trading_pipeline.helpers.artifacts import save_pickle, save_csv
from pairs_trading_pipeline.models.spread import ModelSpread
from pairs_trading_pipeline.models.cointegration import cointegration_from_clusters

import pandas as pd
import numpy as np
import os 
import datetime as dt

ALL_REPRESENTED_SECTORS = [10,15,20,25,30,35,40,45,50,55,60]
PCA_LOADINGS_WRITE_DIR = "Backtest_Intermediate_Data/PCA_Factor_Loadings"
PCA_RETURNS_WRITE_DIR = "Backtest_Intermediate_Data/PCA_Factor_Returns"
PCA_EXPLAINED_VARIANCE_WRITE_DIR = "Backtest_Intermediate_Data/PCA_Explained_Variance_Ratio"
OPTICS_WRITE_DIR = "Backtest_Intermediate_Data/OPTICS_Clusters"
TRUE_COINTEGRATION_WRITE_DIR = "Backtest_Intermediate_Data/CoIntegrated_Pairs"
PORTFOLIO_CANDIDATES_WRITE_DIR = "Backtest_Intermediate_Data/Portfolio_Candidates"
EQUITY_PRICES_READ_DIR = "GICS_Filtered_Equities_Prices"


def compute_portfolio(date, pca_lookback, coint_lookback, z_score_lookback):
    
    pca_end_date = pd.to_datetime(date)
    pca_start_date = pd.to_datetime(time_travel(date, pca_lookback))
    coint_end_date = pca_end_date
    coint_start_date = pd.to_datetime(time_travel(date, coint_lookback))
    
    for sector in ALL_REPRESENTED_SECTORS:
        
        # Applying PCA
        pca_read_path = os.path.join(os.getcwd(), EQUITY_PRICES_READ_DIR, f"GICS_{sector}.pkl")
        df = pd.read_pickle(pca_read_path)
        df = df.loc[pca_start_date:pca_end_date, ]
        factor_returns, factor_loadings, explained_variance_ratio_ = apply_pca(df, remove_cross_sectional_mean=True)
        save_pickle(factor_returns, PCA_RETURNS_WRITE_DIR, f"GICS_{sector}_PCA_FactorReturns_{date}")
        save_pickle(factor_loadings, PCA_LOADINGS_WRITE_DIR, f"GICS_{sector}_PCA_FactorLoadings_{date}")
        save_pickle(explained_variance_ratio_, PCA_EXPLAINED_VARIANCE_WRITE_DIR, f"GICS_{sector}_PCA_ExplainedVarianceRatio_{date}")
        
        # Forming Clusters
        index = find_relevant_PCs(explained_variance_ratio_)
        cluster_pairs = optics_cluster(factor_loadings.iloc[:,0:index])
        save_csv(cluster_pairs, OPTICS_WRITE_DIR, f"GICS_{sector}_clusters_{date}")
        
        # Applying Co-integration
        results = cointegration_from_clusters(os.path.join(os.getcwd(),OPTICS_WRITE_DIR,f"GICS_{sector}_clusters_{date}.csv"), start_date=coint_start_date, end_date=coint_end_date)
        save_csv(results, TRUE_COINTEGRATION_WRITE_DIR, f"GICS_{sector}_cointegrated_pairs_{date}")

    # Finding viable pair candidates across all sectors
    spread = ModelSpread(TRUE_COINTEGRATION_WRITE_DIR)
    portfolio_candidates = spread.compute_spread_stats()
    save_csv(portfolio_candidates, PORTFOLIO_CANDIDATES_WRITE_DIR, f"Portfolio_Candidates_{date}")
    return portfolio_candidates

