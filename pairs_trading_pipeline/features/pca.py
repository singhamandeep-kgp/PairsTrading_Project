import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies PCA to the given dataframe and returns factor returns and factor loadings.
    """
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

    return factor_returns, factor_loadings, explained_variance_ratio_