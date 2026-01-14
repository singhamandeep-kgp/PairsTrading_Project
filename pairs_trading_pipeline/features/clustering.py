import pandas as pd
import numpy as np
from sklearn.cluster import OPTICS
from pairs_trading_pipeline.data.data_processor import DataProcessor


def optics_cluster(
    factor_loadings: pd.DataFrame,
    min_samples: int = 2,
    max_eps: float = np.inf,
    metric: str = "euclidean",
    scale: bool = True) -> pd.Series:
    """
    Run OPTICS clustering on PCA factor loadings and return cluster labels.
    """
    
    if scale:
        processor = DataProcessor(data=factor_loadings)
        factor_loadings = processor.basic_scaler(factor_loadings)

    optics = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric)
    labels = optics.fit_predict(factor_loadings.values)
    cluster_pairs = pd.Series(labels, index=factor_loadings.index, name="optics_label")
    cluster_pairs.sort_values(ascending=False, inplace=True)

    return cluster_pairs