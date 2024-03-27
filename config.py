"""
Configuration Module for Smart Review Analyzer and Recommender (SRAR)

This module contains configuration parameters for the SRAR project, defining essential
settings that control various aspects of data processing and models' behavior.

Attributes:
    DATA_CONFIG (dict): Configuration for data processing.
    MODEL_CONFIG (dict): Configuration for the machine learning models used in the project.
"""

DATA_CONFIG = {
}

MODEL_CONFIG = {
    'sentiment': {
        'test_size': 0.2,                   # Proportion of data reserved for testing
        # 'ngram_range': (1, 2),              # The n-gram range for the TfidfVectorizer
        # 'alpha': 1.0,                       # Smoothing parameter for Naive Bayes
    },
    'thematic': {
        'n_topics': 5,                      # Number of topics to extract
        'n_features': 1000,                 # Number of features for TfidfVectorizer
        'no_top_words': 10,                 # Number of top words to display per topic
    },
}