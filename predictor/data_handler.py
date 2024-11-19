# predictor/data_handler.py

import pandas as pd
import yfinance as yf
from logger import logger  # Import the configured logger

class DataHandler:
    def __init__(self, ticker_symbol: str, period: str = 'max'):
        self.ticker_symbol = ticker_symbol
        self.period = period
        self.data = None
        logger.debug(f"DataHandler initialized with ticker: {ticker_symbol}, period: {period}")

    def fetch_data(self):
        logger.info(f"Fetching data for ticker: {self.ticker_symbol}")
        try:
            tikr = yf.Ticker(self.ticker_symbol)
            self.data = tikr.history(period=self.period)
            self.data.dropna(inplace=True)
            logger.debug(f"Fetched data: {self.data.head()}")
            return self.data
        except Exception as e:
            logger.error(f"Error fetching data for ticker '{self.ticker_symbol}': {e}")
            raise ValueError(f"Error fetching data for ticker '{self.ticker_symbol}': {e}")

    def preprocess_data(self):
        logger.info("Preprocessing data.")
        if self.data is None:
            logger.error("Data not loaded. Call fetch_data() first.")
            raise ValueError("Data not loaded. Call fetch_data() first.")

        # Drop unnecessary columns
        logger.debug("Dropping 'Dividends' column.")
        self.data.drop(['Dividends'], axis=1, inplace=True, errors='ignore')
        # Uncomment if needed
        # logger.debug("Dropping 'Stock Splits' column.")
        # self.data.drop(['Stock Splits'], axis=1, inplace=True, errors='ignore')

        # Create target variable
        logger.debug("Creating 'Tomorrow' and 'Target' columns.")
        self.data['Tomorrow'] = self.data['Close'].shift(-1)
        self.data['Target'] = (self.data['Tomorrow'] > self.data['Close']).astype(int)

        # Additional Predictors
        horizons = [2, 5, 60]
        new_predictors = []
        for horizon in horizons:
            logger.debug(f"Creating predictors for horizon: {horizon}")
            rolling_avg = self.data['Close'].rolling(window=horizon, min_periods=1).mean().shift(1)
            ratio_column = f"Close_Ratio_{horizon}"
            self.data[ratio_column] = rolling_avg / self.data['Close']
            logger.debug(f"Created {ratio_column}")

            trend_column = f"Trend_{horizon}"
            self.data[trend_column] = self.data['Target'].shift(1).rolling(window=horizon, min_periods=1).sum()
            logger.debug(f"Created {trend_column}")

            new_predictors.extend([ratio_column, trend_column])

        # Define predictors
        self.predictors = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.predictors.extend(new_predictors)
        logger.debug(f"Total predictors: {self.predictors}")

        # Drop rows with NaN values in predictors or target
        logger.debug("Dropping rows with NaN values in predictors or target.")
        self.data.dropna(subset=self.predictors + ['Target'], inplace=True)
        self.data = self.data.to_period('D')
        logger.debug(f"Data after preprocessing: {self.data.head()}")

        return self.data, self.predictors
