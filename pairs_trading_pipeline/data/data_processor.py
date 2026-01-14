import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataProcessor:

    def __init__(self, data: pd.DataFrame) -> None: 
        self.data = data

    def basic_scaler(self, df=None) -> pd.DataFrame:
        """
        Z-Scoring
        """

        if df is None:
            df = self.data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        return df
    
    def scaler(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        z scoring but first removes cross-sectional mean return from each date across all columns in dataframe
        """
        
        if df is None:
            df = self.data

        df = self.remove_cross_sectional_mean(df)
        df = df.pct_change()
        df.dropna(inplace = True)
        df = self.basic_scaler(df)

        return df
    
    def remove_cross_sectional_mean(self, df: pd.DataFrame = None) ->pd.DataFrame:
        """
        Removes the cross-sectional mean return from each date across all columns in dataframe
        """
        if df is None:
            df = self.data
        return df.sub(df.mean(axis=1), axis=0)
