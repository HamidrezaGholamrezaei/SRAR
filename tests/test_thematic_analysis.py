"""
Unit Test Module for Thematic Analysis in Smart Review Analyzer and Recommender (SRAR)

This module contains a suite of unit tests designed to validate the functionality and reliability 
of the thematic analysis components within the SRAR project. 
"""

import unittest
from unittest.mock import patch, call, MagicMock
import os
import numpy as np
from src.thematic_analysis.thematic_analyzer import ThematicAnalyzer


class TestThematicAnalyzer(unittest.TestCase):
    def setUp(self):
        self.config_mock = MagicMock()
        self.config_mock.evaluation_directory = './evaluations'
        self.config_mock.dataset = 'test_dataset'
        self.documents = ['Sample text 1', 'Sample text 2']
        self.analyzer = ThematicAnalyzer(self.config_mock, self.documents)

    def test_execute_model_workflow(self):
        with patch.object(self.analyzer, 'fit_transform', MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))) as mock_fit_transform, \
            patch.object(self.analyzer, 'display_topics', MagicMock()) as mock_display_topics, \
            patch.object(self.analyzer, 'visualize_topics', MagicMock()) as mock_visualize_topics, \
            patch.object(self.analyzer, 'compute_topic_coherence', MagicMock()) as mock_compute_topic_coherence, \
            patch.object(self.analyzer.tfidf_vectorizer, 'get_feature_names_out', MagicMock(return_value=['feature1', 'feature2'])) as mock_get_feature_names_out, \
            patch('src.thematic_analysis.thematic_analyzer.logging.info') as mock_logging_info:

            self.analyzer.execute_model_workflow()

            mock_fit_transform.assert_called_once()
            mock_display_topics.assert_called_once()
            mock_visualize_topics.assert_called_once()
            mock_compute_topic_coherence.assert_called_once()
            mock_get_feature_names_out.assert_called_once()

            expected_calls = [call("Starting thematic analysis..."), call("Thematic analysis completed.")]
            mock_logging_info.assert_has_calls(expected_calls, any_order=False)

    def test_fit_transform_success(self):
        with patch.object(self.analyzer.tfidf_vectorizer, 'fit_transform', MagicMock(return_value='tfidf_matrix')) as mock_fit_transform, \
             patch.object(self.analyzer.nmf_model, 'fit_transform', MagicMock(return_value='W_matrix')) as mock_nmf_fit, \
             patch.object(self.analyzer.nmf_model, 'components_', 'H_matrix', create=True), \
             patch('src.data_processing.data_handler.logging.error') as mock_log_error:
            
            W, H, tfidf_matrix = self.analyzer.fit_transform()
            
            mock_fit_transform.assert_called_once_with(self.documents)
            mock_nmf_fit.assert_called_once_with('tfidf_matrix')
            self.assertEqual(H, 'H_matrix')
            self.assertIsNotNone(W)
            self.assertIsNotNone(tfidf_matrix)
            mock_log_error.assert_not_called()

    def test_fit_transform_exception(self):
        with patch.object(self.analyzer.tfidf_vectorizer, 'fit_transform', side_effect=Exception("Test Error")), \
            patch('src.data_processing.data_handler.logging.error') as mock_log_error:
            with self.assertRaises(Exception) as context:
                self.analyzer.fit_transform()
            self.assertTrue("Test Error" in str(context.exception))
            mock_log_error.assert_called_once_with("Failed to fit_transform documents: Test Error")

    def test_display_topics(self):
        nmf_model_mock = MagicMock()
        nmf_model_mock.components_ = np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]])
        self.analyzer.nmf_model = nmf_model_mock
        feature_names = ['word1', 'word2', 'word3']

        with patch('src.thematic_analysis.thematic_analyzer.logging.info') as mock_log_info:
            self.analyzer.display_topics(feature_names)
            expected_calls = [
                call("Top words per topic:"),
                call("Topic #0: word3"),
                call("Topic #1: word1"),
            ]
            mock_log_info.assert_has_calls(expected_calls, any_order=False)

    def test_compute_topic_coherence(self):
        with patch.object(self.analyzer, 'prepare_corpus', return_value=([], {}, [])) as mock_prepare_corpus, \
             patch('src.thematic_analysis.thematic_analyzer.CoherenceModel') as mock_coherence_model, \
             patch('src.thematic_analysis.thematic_analyzer.logging.info') as mock_log_info:
            mock_coherence_instance = mock_coherence_model.return_value
            mock_coherence_instance.get_coherence.return_value = 0.75
            self.analyzer.compute_topic_coherence(MagicMock(name='H'))
            mock_prepare_corpus.assert_called_once()
            mock_coherence_instance.get_coherence.assert_called_once()
            mock_log_info.assert_called_with("Average Topic Coherence: 0.7500")

    def test_visualize_topics(self):
        H_mock = np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]])
        feature_names_mock = ['word1', 'word2', 'word3']
        with patch('src.thematic_analysis.thematic_analyzer.plt.savefig') as mock_savefig:
            self.analyzer.visualize_topics(H_mock, feature_names_mock)
            expected_figure_path = os.path.join(self.analyzer.config.evaluation_directory, f'topic_word_heatmap_{self.analyzer.config.dataset}.png')
            mock_savefig.assert_called_with(expected_figure_path, bbox_inches='tight')

    def test_prepare_corpus_failure(self):
        analyzer = ThematicAnalyzer(self.config_mock, [''])
        with patch('src.thematic_analysis.thematic_analyzer.simple_preprocess', side_effect=Exception("Test Exception")), \
             patch('src.thematic_analysis.thematic_analyzer.logging.error') as mock_log_error:
            with self.assertRaises(Exception):
                analyzer.prepare_corpus()
            mock_log_error.assert_called_with("Failed to prepare corpus: Test Exception")

if __name__ == '__main__':
    unittest.main()
