"""
Data Processing Package for Smart Review Analyzer and Recommender (SRAR)

This package contains modules and functionalities for loading, preprocessing, and handling 
data required by the SRAR project. It is designed to facilitate various data operations 
and transformations to prepare the data for analysis and insights extraction.
"""

from .data_handler import load_data

__all__ = ["load_data"]
