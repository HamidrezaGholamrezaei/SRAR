"""
Sentiment Analyzer Module for Smart Review Analyzer and Recommender (SRAR)

This module implements the SentimentAnalyzer class, which leverages a Naive Bayes classifier
for sentiment analysis. It utilizes a TF-IDF vectorizer for feature extraction from text data,
facilitating the classification of sentiments into predefined categories such as 'positive' or 'negative'.
"""

import logging
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class SentimentAnalyzer:
    """
    A class for constructing, training, and evaluating a sentiment analysis model based on TfidfVectorizer and MultinomialNB.

    Attributes:
        config (argparse.Namespace): Configuration parameters.
        sequences (list): List of text sequences for training.
        labels (list): List of sentiment labels corresponding to the text sequences.

    Methods:
        execute_model_workflow(): Executes the training or prediction workflow.
        train(): Trains the model on the provided dataset and evaluates its performance
        predict(): Predicts sentiment for a given text sequence.
        save_model(): Saves the trained model to a file.
        load_model(): Loads a pre-trained model.
        save_evaluation_results(results: dict): Saves the evaluation's results to a file.
    """

    def __init__(self, config, sequences, labels):
        self.config = config
        self.sequences = sequences
        self.labels = labels
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        logging.info("SentimentAnalyzer initialized successfully.")

    def execute_model_workflow(self):
        if self.config.action == 'train':
            self.train()
            if self.config.text is not None:
                logging.info("Continue with predictions...")
                sentiment = self.predict()
                logging.info(f"Predicted sentiment: {sentiment}")
        elif self.config.action == 'predict':
            if self.config.text is None:
                logging.error("No text provided for prediction.")
                raise ValueError("Text required for prediction when action is 'predict'.")
            self.load_model()
            logging.info("Model loaded successfully for predictions.")
            sentiment = self.predict()
            logging.info(f"Predicted sentiment: {sentiment}")

    def train(self):
        logging.info("Training the model...")
        X_train, X_test, y_train, y_test = train_test_split(self.sequences, self.labels, test_size=self.config.test_size, random_state=self.config.seed)
        logging.info("Starting model training...")
        self.model.fit(X_train, y_train)
        self.save_model()
        logging.info("Model training completed. Proceeding with evaluation...")

        # Evaluating the model on the test set
        y_pred = self.model.predict(X_test)
        evaluation_results = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred, average='weighted')
        }

        logging.info(f"Evaluation Metrics: {evaluation_results}")

    def predict(self):
        logging.info(f"Provided text for prediction: {self.config.text}")
        prediction = self.model.predict([self.config.text])[0]
        sentiment = "positive" if prediction > .5 else "negative"
        return sentiment

    def save_model(self):
        try:
            joblib.dump(self.model, self.config.model_path)
            logging.info(f"Model successfully saved to {self.config.model_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def load_model(self):
        try:
            self.model = joblib.load(self.config.model_path)
            logging.info(f"Model successfully loaded from {self.config.model_path}")
        except FileNotFoundError as e:
            logging.error(f"Error loading model: {e}")
            raise
