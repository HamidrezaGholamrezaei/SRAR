"""
Main Module for Smart Review Analyzer and Recommender (SRAR)

This module serves as the entry point for the SRAR project. It initializes the application's 
configuration, and orchestrates the overall analysis workflow.

Author: Hamidreza Gholamrezaei
Created: March 2024
"""

import os
import logging
import datetime
from src.data_processing.data_handler import load_data
from src.sentiment_analysis.sentiment_analyzer import SentimentAnalyzer
from src.thematic_analysis.thematic_analyzer import ThematicAnalyzer
import utils


def setup_logging():
    """
    Sets up logging configuration for the project.
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_directory = 'logs'
    log_filename = os.path.join(log_directory, f"log_{timestamp}.log")

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=log_filename,
                        filemode='w')
    logging.getLogger('gensim').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger().addHandler(logging.StreamHandler())


def perform_sentiment_analysis(config, sequences, labels):
    sentiment_analyzer = SentimentAnalyzer(config, sequences, labels)
    sentiment_analyzer.execute_model_workflow()

def perform_thematic_analysis(config, sequences):
    thematic_analyzer = ThematicAnalyzer(config, sequences)
    thematic_analyzer.execute_model_workflow()


def main():
    config = utils.build_config()
    if not config:
        logging.error("Configuration Error.")
        return
    
    try:
        sequences, labels = load_data(config.dataset)
        logging.info("Data successfully loaded and preprocessed.")
        
        perform_sentiment_analysis(config, sequences, labels)
        perform_thematic_analysis(config, sequences)
        
    except Exception as e:
        logging.error("An error occurred during the analysis process: %s", e)


if __name__ == '__main__':
    setup_logging()
    main()
