# predictor/model_trainer.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from logger import logger  # Import the configured logger

class ModelTrainer:
    def __init__(self, predictors: list):
        self.predictors = predictors
        self.model = RandomForestClassifier(
            n_estimators=200,
            min_samples_split=50,
            random_state=1,
            max_features=None,
            max_depth=6
        )
        self.scaler = StandardScaler()
        logger.debug(f"ModelTrainer initialized with predictors: {self.predictors}")

    def train_test_split_data(self, data: pd.DataFrame):
        logger.info("Splitting data into training and testing sets.")
        train = data.loc[(data.index > '2009-02-01') & (data.index < '2019-12-01')]
        test = data.loc[data.index > '2022-09-01']
        logger.debug(f"Training data size: {len(train)}")
        logger.debug(f"Testing data size: {len(test)}")
        return train, test

    def scale_data(self, train: pd.DataFrame, test: pd.DataFrame):
        logger.info("Scaling data.")
        self.scaler.fit(train[self.predictors])
        logger.debug(f"Scaler mean: {self.scaler.mean_}")
        logger.debug(f"Scaler scale: {self.scaler.scale_}")

        train_scaled = train.copy()
        test_scaled = test.copy()

        train_scaled[self.predictors] = self.scaler.transform(train[self.predictors])
        test_scaled[self.predictors] = self.scaler.transform(test[self.predictors])

        logger.debug("Data scaling completed.")
        return train_scaled, test_scaled

    def train_model(self, train_scaled: pd.DataFrame):
        logger.info("Training the Random Forest model.")
        self.model.fit(train_scaled[self.predictors], train_scaled['Target'])
        logger.debug("Model training completed.")
        return self.model

    def evaluate_model(self, test_scaled: pd.DataFrame):
        logger.info("Evaluating the model.")
        predictions = self.model.predict(test_scaled[self.predictors])
        predictions = pd.Series(predictions, index=test_scaled.index)
        precision = precision_score(test_scaled['Target'], predictions,zero_division=1)
        logger.debug(f"Precision score: {precision}")
        return precision, predictions
