from pairs_trading_pipeline.data.data_processor import DataProcessor
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_pca(df: pd.DataFrame, remove_cross_sectional_mean = True) -> pd.DataFrame:

    if remove_cross_sectional_mean:
        dp = DataProcessor(df)
        df = dp.scaler()
        logger.info("Cross-sectional mean removed.")

    pca = PCA()
    pca_data = pca.fit_transform(df)
    explained_variance_ratio_ = pca.explained_variance_ratio_

    factor_returns = pd.DataFrame(
        pca_data,
        index=df.index,
        columns=[f"PC{i+1}" for i in range(pca_data.shape[1])]
    )

    factor_loadings = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=factor_returns.columns
    )
    logger.info("PCA Computed")
    return factor_returns, factor_loadings, explained_variance_ratio_

def find_relevant_PCs(explained_variance_ratio, benchmark = .50):
    """ First index at which PCs explain benchmark level of variation in sector-wise equity returns"""
    logger.info("Finding relevant PCs for benchmark: %f", benchmark)

    arr = np.asarray(explained_variance_ratio, dtype=float)
    csum = np.cumsum(arr)
    mask = csum > benchmark
    if not mask.any():
        logger.warning("No PCs found that explain more than the benchmark variance.")
        return None  
    i = mask.argmax()  
    logger.info("Relevant PC index: %d", i)
    return (i + 1)