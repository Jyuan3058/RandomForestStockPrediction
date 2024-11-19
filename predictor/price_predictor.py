# predictor/price_predictor.py

from .data_handler import DataHandler
from .model_trainer import ModelTrainer
from .backtester import Backtester
from sklearn.metrics import precision_score
from logger import logger  # Import the configured logger

class PricePredictor:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data_handler = DataHandler(ticker)
        self.model_trainer = None
        self.backtester = None
        logger.debug(f"PricePredictor initialized for ticker: {ticker}")

    def run(self):
        logger.info(f"Starting prediction workflow for ticker: {self.ticker}")
        # Data Handling
        logger.debug("Fetching data.")
        data = self.data_handler.fetch_data()
        logger.debug("Preprocessing data.")
        processed_data, predictors = self.data_handler.preprocess_data()

        # Model Training
        logger.debug("Initializing ModelTrainer.")
        self.model_trainer = ModelTrainer(predictors)
        logger.debug("Splitting data into train and test sets.")
        train, test = self.model_trainer.train_test_split_data(processed_data)
        logger.debug("Scaling data.")
        train_scaled, test_scaled = self.model_trainer.scale_data(train, test)
        logger.debug("Training the model.")
        self.model_trainer.train_model(train_scaled)
        logger.debug("Evaluating the model.")
        initial_precision, initial_predictions = self.model_trainer.evaluate_model(test_scaled)
        logger.info(f"Initial Precision Score: {initial_precision:.4f}")

        # Backtesting
        logger.debug("Initializing Backtester.")
        self.backtester = Backtester(self.model_trainer.model, predictors, self.model_trainer.scaler)
        logger.debug("Running backtest.")
        backtest_results = self.backtester.backtest(processed_data)

        # Calculate Backtest Precision
        logger.debug("Calculating backtest precision score.")
        backtest_precision = precision_score(backtest_results['Target'], backtest_results['Predictions'])
        logger.info(f"Backtest Precision Score: {backtest_precision:.4f}")

        # Value Counts of Predictions
        prediction_counts = backtest_results['Predictions'].value_counts()
        logger.debug(f"Prediction Counts: {prediction_counts.to_dict()}")

        return {
            'initial_precision': initial_precision,
            'backtest_precision': backtest_precision,
            'prediction_counts': prediction_counts.to_dict()
        }
