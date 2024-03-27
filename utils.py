"""
Utility Module for Smart Review Analyzer and Recommender (SRAR)

This module provides utility functions to assist with command-line interface setup,
environment configuration, and other supporting tasks for the SRAR project.
"""

import argparse
import logging
import os
import random
import config

def build_config():
    """
    Constructs the configurations for the SRAR project.

    Returns:
        argparse.Namespace: Configuration settings.
    """

    parser = argparse.ArgumentParser(description="Configure the Smart Review Analyzer and Recommender (SRAR) project.")
    parser.add_argument('--action', choices=['train', 'predict'], default='train', help="Specifies the action to perform: 'train' a new model or 'predict' using an existing model.")
    parser.add_argument('--dataset', default='imdb', help="Specifies the dataset name. Default is 'imdb'.")
    parser.add_argument('--text', type=str, help="Text for sentiment prediction. Required if action is 'predict'.")
    parser.add_argument('--result_dir', default='./results', help="Directory to save results.")
    parser.add_argument('--seed', type=int, default=100, help="Random seed for reproducibility.")
    parser.add_argument('--test_size', type=float, default=config.MODEL_CONFIG['sentiment']['test_size'], help="Proportion of data reserved for testing in sentiment analysis")
    parser.add_argument('--n_topics', type=int, default=config.MODEL_CONFIG['thematic']['n_topics'], help="Number of topics for thematic analysis.")
    parser.add_argument('--n_features', type=int, default=config.MODEL_CONFIG['thematic']['n_features'], help="Number of features for TfidfVectorizer for thematic analysis.")
    parser.add_argument('--no_top_words', type=int, default=config.MODEL_CONFIG['thematic']['no_top_words'], help="Number of top words to display per topic.")

    args = parser.parse_args()
    handle_missing_arguments(args)
    setup_directories(args)
    set_random_seed(args.seed)
    logging.debug("Configuration settings: %s", args)
    return args

def handle_missing_arguments(args):
    """
    Validates and handles missing arguments based on the specified action.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    
    if args.action == 'predict' and not args.text:
        logging.error("Text is required for prediction.")
        raise ValueError("Text required for prediction when action is 'predict'.")

def setup_directories(args):
    """
    Validates and prepares necessary directories based on provided configurations.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    model_directory = os.path.join(args.result_dir, 'models')
    args.evaluation_directory = os.path.join(args.result_dir, 'evaluations')
    vocab_directory = os.path.join('data', 'vocab')
    args.model_path = os.path.join(model_directory, f'model_{args.dataset}.pth')

    for directory in [model_directory, args.evaluation_directory, vocab_directory]:
        os.makedirs(directory, exist_ok=True)

    if args.action == 'predict' and not os.path.exists(args.model_path):
        logging.warning(f"Model for '{args.dataset}' not found for prediction. Switching to 'train' action.")
        args.action = 'train'

    if args.action == 'train':
        dataset_file_path = os.path.join('data', f"{args.dataset}.csv")
        if not os.path.exists(dataset_file_path):
            logging.error(f"The dataset file for '{args.dataset}' does not exist at {dataset_file_path}")
            raise FileNotFoundError(f"Dataset file not found: {dataset_file_path}")

def set_random_seed(seed):
    """
    Sets the random seed for reproducibility.

    Parameters:
        seed (int): The seed value for random number generators.
    """

    random.seed(seed)
    logging.debug(f"Random seed set to {seed}")
