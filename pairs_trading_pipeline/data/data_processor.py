import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:

    def __init__(self, data: pd.DataFrame) -> None: 
        self.data = data

    def basic_scaler(self, df=None) -> pd.DataFrame:
        """
        Z-Scoring
        """
        if df is None:
            df = self.data
        logger.info("Applying basic scaler to data of shape %s", df.shape)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        logger.info("Basic scaling complete.")
        return df

    def scaler(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        z scoring but first removes cross-sectional mean return from each date across all columns in dataframe
        """
        if df is None:
            df = self.data

        df = self.remove_cross_sectional_mean(df)
        df = df.pct_change()
        df.dropna(inplace=True)
        df = self.basic_scaler(df)
        logger.info("Scaler process complete.")

        return df

    def remove_cross_sectional_mean(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Removes the cross-sectional mean return from each date across all columns in dataframe
        """
        if df is None:
            df = self.data
        logger.info("Removing cross-sectional mean from data of shape %s", df.shape)
        return df.sub(df.mean(axis=1), axis=0)
