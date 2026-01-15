import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from pairs_trading_pipeline.data.data_processor import DataProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def optics_cluster(
    factor_loadings: pd.DataFrame,
    min_samples: int = 2,
    max_eps: float = np.inf,
    metric: str = "euclidean",
    scale: bool = True) -> pd.Series:

    logger.info("Starting OPTICS clustering with min_samples=%d, max_eps=%f, metric=%s, scale=%s", min_samples, max_eps, metric, scale)

    if scale:
        processor = DataProcessor(data=factor_loadings)
        factor_loadings = processor.basic_scaler(factor_loadings)

    optics = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric)

    labels = optics.fit_predict(factor_loadings.values)
    logger.info("OPTICS clustering complete. Number of clusters found: %d", len(set(labels)) - (1 if -1 in labels else 0))

    cluster_pairs = pd.Series(labels, index=factor_loadings.index, name="optics_label")
    cluster_pairs.sort_values(ascending=False, inplace=True)

    return cluster_pairs