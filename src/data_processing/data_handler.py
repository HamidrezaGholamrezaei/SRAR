"""
Data Handling Module for Smart Review Analyzer and Recommender (SRAR)

This module manages data operations within the SRAR project. It includes functionalities 
for loading datasets, performing data preprocessing, and preparing data for further 
analysis and processing.
"""

import logging
import os
import pandas as pd
from .data_preprocesser import prepare_data_for_analysis


def load_data(data_name):
    """
    Load data from a CSV file based on the provided dataset name.

    Parameters:
        data_name (str): Identifier for the dataset.

    Returns:
        DataFrame: Pandas DataFrame containing the loaded data.
    """

    file_name = f"{data_name}.csv"
    file_path = os.path.join("./data", file_name)

    if not os.path.exists(file_path):
        error_msg = f"Data file '{file_name}' not found in data directory."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    logging.info(f"Loading data from {file_name}")
    raw_data = pd.read_csv(file_path)
    return preprocess_data(data_name, raw_data)

def preprocess_data(data_name, data):
    """
    Preprocess raw data based on the dataset type.

    Parameters:
        data_name (str): Identifier for the dataset.
        data (DataFrame): The raw data as a DataFrame.

    Returns:
        DataFrame: A DataFrame containing processed texts and labels.
    """

    preprocess_func = {
        "imdb": preprocess_imdb_data,
        "financial": preprocess_financial_data
    }.get(data_name)

    if preprocess_func:
        return preprocess_func(data)
    else:
        logging.warning(f"Unknown dataset: {data_name}")
        raise ValueError(f"Unknown dataset: {data_name}")

def preprocess_imdb_data(data):
    """
    Preprocess IMDb review data.

    Parameters:
        data (DataFrame): IMDb review data.

    Returns:
        DataFrame: A DataFrame containing processed texts and labels.
    """

    logging.debug("Processing imdb data.")
    return prepare_data_for_analysis(data, 'review', 'sentiment')

def preprocess_financial_data(data):
    """
    Preprocess financial phrases data.

    Parameters:
        data (DataFrame): financial phrases data.

    Returns:
        DataFrame: A DataFrame containing processed texts and labels.
    """

    logging.debug("Processing financial data.")

    if not {'phrase', 'sentiment'}.issubset(data.columns):
        raise KeyError("Financial data must contain 'phrase' and 'sentiment' columns.")
    
    return prepare_data_for_analysis(data, 'phrase', 'sentiment')
