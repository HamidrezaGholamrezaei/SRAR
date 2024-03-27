"""
Unit Test Module for Sentiment Analysis in Smart Review Analyzer and Recommender (SRAR)

This module contains a suite of unit tests designed to validate the functionality and reliability 
of the sentiment analysis components within the SRAR project. 
"""

import unittest
from unittest.mock import patch, MagicMock
from src.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.config_mock = MagicMock()
        self.config_mock.model_path = './model.pth'
        self.config_mock.results_path = './results.txt'
        self.config_mock.test_size = 0.1
        self.config_mock.seed = 42
        self.config_mock.text = 'Sample text for prediction'
        self.config_mock.action = 'train'
        self.sequences = ['Sample text 1', 'Sample text 2']
        self.labels = [0, 1]

    def test_execute_model_workflow_train(self):
        self.config_mock.action = 'train'
        self.config_mock.text = None
        analyzer = SentimentAnalyzer(self.config_mock, self.sequences, self.labels)
        with patch.object(analyzer, 'train', MagicMock()) as mock_train, \
            patch.object(analyzer, 'predict', MagicMock()) as mock_predict:
            analyzer.execute_model_workflow()
            mock_train.assert_called_once()
            mock_predict.assert_not_called()

    def test_execute_model_workflow_predict(self):
        self.config_mock.action = 'predict'
        analyzer = SentimentAnalyzer(self.config_mock, self.sequences, self.labels)
        with patch.object(analyzer, 'load_model', MagicMock()) as mock_load_model, \
            patch.object(analyzer, 'predict', MagicMock()) as mock_predict:
            analyzer.execute_model_workflow()
            mock_load_model.assert_called_once()
            mock_predict.assert_called_once()

    def test_predict_valid_text(self):
        analyzer = SentimentAnalyzer(self.config_mock, self.sequences, self.labels)
        with patch.object(analyzer.model, 'predict', return_value=[0]):
            prediction = analyzer.predict()
            self.assertIn(prediction, ['positive', 'negative'])

    def test_predict_empty_text(self):
        self.config_mock.text = ''
        analyzer = SentimentAnalyzer(self.config_mock, self.sequences, self.labels)
        with self.assertRaises(ValueError):
            analyzer.predict()

    def test_train_empty_dataset(self):
        analyzer = SentimentAnalyzer(self.config_mock, [], [])
        with self.assertRaises(ValueError):
            analyzer.train() 

    def test_save_model(self):
        analyzer = SentimentAnalyzer(self.config_mock, self.sequences, self.labels)
        with patch('joblib.dump') as mock_joblib_dump:
            analyzer.save_model()
            mock_joblib_dump.assert_called_once()

    def test_save_model_invalid_path(self):
        analyzer = SentimentAnalyzer(self.config_mock, self.sequences, self.labels)
        with patch('joblib.dump') as mock_joblib_dump, \
            patch('logging.error') as mock_log_error:
            mock_joblib_dump.side_effect = OSError("Cannot write to file")
            with self.assertRaises(OSError):
                analyzer.save_model()
            mock_log_error.assert_called_with("Error saving model: Cannot write to file")

    def test_load_model(self):
        analyzer = SentimentAnalyzer(self.config_mock, self.sequences, self.labels)
        with patch('joblib.load') as mock_joblib_load:
            mock_joblib_load.return_value = MagicMock()
            analyzer.load_model()
            mock_joblib_load.assert_called_once_with(self.config_mock.model_path)

    def test_load_model_invalid_path(self):
        analyzer = SentimentAnalyzer(self.config_mock, self.sequences, self.labels)
        with patch('joblib.load') as mock_joblib_load, \
            patch('logging.error') as mock_log_error:
            mock_joblib_load.side_effect = FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                analyzer.load_model()
            mock_log_error.assert_called_with("Error loading model: ")    
    

if __name__ == '__main__':
    unittest.main(verbosity=2)
