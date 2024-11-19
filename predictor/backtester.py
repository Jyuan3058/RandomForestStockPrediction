# predictor/backtester.py

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from logger import logger  # Import the configured logger

class Backtester:
    def __init__(self, model, predictors: list, scaler: StandardScaler):
        self.model = model
        self.predictors = predictors
        self.scaler = scaler
        logger.debug(f"Backtester initialized with predictors: {self.predictors}")

    def predict(self, train_scaled: pd.DataFrame, test_scaled: pd.DataFrame):
        logger.info("Making predictions on training data.")
        self.model.fit(train_scaled[self.predictors], train_scaled['Target'])

        # Training predictions for accuracy
        logger.debug("Predicting on training data.")
        train_preds_proba = self.model.predict_proba(train_scaled[self.predictors])[:, 1]
        train_preds = (train_preds_proba >= 0.6).astype(int)
        train_accuracy = accuracy_score(train_scaled['Target'], train_preds)
        logger.debug(f"Training Accuracy: {train_accuracy:.4f}")

        # Test predictions
        logger.debug("Predicting on test data.")
        test_preds_proba = self.model.predict_proba(test_scaled[self.predictors])[:, 1]
        test_preds = (test_preds_proba >= 0.6).astype(int)
        test_accuracy = accuracy_score(test_scaled['Target'], test_preds)
        logger.debug(f"Testing Accuracy: {test_accuracy:.4f}")

        # Combine predictions with actual targets
        logger.debug("Combining predictions with actual targets.")
        combined = pd.concat([
            test_scaled['Target'],
            pd.Series(test_preds, index=test_scaled.index, name='Predictions')
        ], axis=1)
        return combined

    def backtest(self, data: pd.DataFrame, start: int = 1000, step: int = 250):
        all_predictions = []
        logger.info(f"Starting backtest with start={start} and step={step}.")
        logger.debug(f"Predictors used: {self.predictors}")
        logger.debug(f"Shape of Data: {data.shape[0]} rows")

        for i in range(start, data.shape[0], step):
            logger.debug(f"Backtesting window: {i} to {i+step}")
            train = data.iloc[0:i].copy()
            test = data.iloc[i:i+step].copy()

            # Drop rows with NaNs
            logger.debug("Dropping rows with NaNs in test data.")
            test.dropna(subset=self.predictors + ['Target'], inplace=True)

            # Scale data
            logger.debug("Fitting scaler on training data.")
            self.scaler.fit(train[self.predictors])
            train_scaled = train.copy()
            test_scaled = test.copy()
            train_scaled[self.predictors] = self.scaler.transform(train[self.predictors])
            test_scaled[self.predictors] = self.scaler.transform(test[self.predictors])

            # Make predictions
            logger.debug("Making predictions for the current window.")
            predictions = self.predict(train_scaled, test_scaled)
            all_predictions.append(predictions)
            logger.debug("Predictions added to all_predictions.")

        combined_predictions = pd.concat(all_predictions)
        logger.info("Backtesting completed.")
        return combined_predictions
