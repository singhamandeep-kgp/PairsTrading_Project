import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os  

class DataProcessor:

    def __init__(self, data: pd.DataFrame) -> None: 
        self.data = data

    def basic_scaler(self, df=None) -> pd.DataFrame:
        if df is None:
            df = self.data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        return df
    
    def scaler(self) -> pd.DataFrame:
        
        df = self.remove_cross_sectional_mean()
        df = df.pct_change()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        df.dropna(inplace = True)
        return df

    def apply_pca(self, df = None) -> pd.DataFrame:
        if df is None:
            df = self.data
        pca = PCA()
        pca_data = pca.fit_transform(df)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        return pd.DataFrame(pca_data, index=df.index)
    
    def remove_cross_sectional_mean(self) ->pd.DataFrame:
        """
        Removes the cross-sectional mean return from each date across all columns in dataframe
        """
        return self.data.sub(self.data.mean(axis=1), axis=0)
