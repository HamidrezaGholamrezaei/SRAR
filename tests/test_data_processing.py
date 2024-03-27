"""
Unit Test Module for Data Processing in Smart Review Analyzer and Recommender (SRAR)

This module contains a suite of unit tests designed to validate the data processing functionalities 
of the SRAR project. It ensures the correctness, efficiency, and error handling of data loading, 
preprocessing, and various data scenarios.
"""

import unittest
import pandas as pd
import time
from unittest.mock import patch, ANY
from src.data_processing.data_handler import load_data, preprocess_data, preprocess_imdb_data, preprocess_financial_data
from src.data_processing.data_preprocesser import clean_text, prepare_data_for_analysis


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.imdb_sample_data = pd.DataFrame({
            'review': [
                'This is a positive review. Absolutely loved it! <br/>',
                'This was a terrible movie. Never watching again!'
            ],
            'sentiment': ['positive', 'negative']
        })

        self.financial_sample_data = pd.DataFrame({
            'phrase': [
                'Earnings exceeded expectations.', 
                'Losses were higher than last quarter.'
            ],
            'sentiment': ['positive', 'negative']
        })
        self.mock_text = "This is a <b>bold</b> statement! <html> and this @ URL: http://example.com should be removed."
        self.cleaned_text = "this bold statement and this url should removed"

    def test_load_data_valid_file(self):
        with patch('src.data_processing.data_handler.os.path.exists', return_value=True), \
             patch('src.data_processing.data_handler.pd.read_csv', return_value=pd.DataFrame({'dummy': [1]})) as mock_read_csv, \
             patch('src.data_processing.data_handler.preprocess_data', return_value=pd.DataFrame({'processed': [1]})) as mock_preprocess:
            data = load_data('valid_dataset')
            mock_read_csv.assert_called_once()
            mock_preprocess.assert_called_once_with('valid_dataset', ANY)
            self.assertIsInstance(data, pd.DataFrame)

    def test_load_data_invalid_file(self):
        with patch('src.data_processing.data_handler.os.path.exists', return_value=False) as mock_exists, \
            patch('src.data_processing.data_handler.logging.error') as mock_log_error:
            with self.assertRaises(FileNotFoundError):
                load_data('invalid_dataset')
            mock_log_error.assert_called_with("Data file 'invalid_dataset.csv' not found in data directory.")

    def test_preprocess_imdb_data(self):
        sequences, labels = preprocess_imdb_data(self.imdb_sample_data)
        self.assertEqual(len(sequences), len(self.imdb_sample_data))
        self.assertEqual(len(labels), len(self.imdb_sample_data))

    def test_preprocess_financial_data_valid(self):
        sequences, labels = preprocess_financial_data(self.financial_sample_data)
        self.assertEqual(len(sequences), len(self.financial_sample_data))
        self.assertEqual(len(labels), len(self.financial_sample_data))

    def test_clean_text(self):
        cleaned = clean_text(self.mock_text)
        self.assertEqual(cleaned, self.cleaned_text)

    def test_data_integration(self):
        processed_texts, labels = preprocess_data('imdb', self.imdb_sample_data)
        self.assertEqual(len(processed_texts), len(self.imdb_sample_data))
        self.assertEqual(len(labels), len(self.imdb_sample_data['sentiment']))
        self.assertTrue(all(isinstance(text, str) for text in processed_texts))
        self.assertTrue(all(label in [0, 1] for label in labels))
        expected_labels = [1, 0]
        self.assertEqual(list(labels), expected_labels)
    
    def test_missing_required_columns(self):
        incomplete_data = self.financial_sample_data.drop(columns=['sentiment'])
        with self.assertRaises(KeyError) as context:
            prepare_data_for_analysis(incomplete_data, 'review', 'sentiment')
        self.assertIn("Missing required columns", str(context.exception))

    @patch('src.data_processing.data_preprocesser.clean_text')
    def test_error_during_preprocessing(self, mock_clean_text):
        mock_clean_text.side_effect = Exception("Unexpected error")
        with self.assertRaises(Exception):
            prepare_data_for_analysis(self.financial_sample_data, 'text', 'sentiment')

    def test_performance(self):
        start_time = time.time()
        clean_text(self.mock_text * 1000)
        end_time = time.time()
        self.assertTrue((end_time - start_time) < 1, "Text cleaning should be fast.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
