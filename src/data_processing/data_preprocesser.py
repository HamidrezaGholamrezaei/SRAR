"""
Data Preprocessing Module for Smart Review Analyzer and Recommender (SRAR)

This module provides functions for text cleaning, tokenization, stemming, 
vocabulary building, and preparing data for analysis. It ensures the text data 
is in a suitable format for desired tasks.
"""

import re
import logging
import contractions
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def replace_numbers(text):
    """
    Replace all numeric values in the text with a placeholder token '<NUM>'.
    """

    return re.sub(r'\b\d+\b', '<NUM>', text)

def is_repetitive(token, threshold=3):
    """
    Check if a token consists of the same character repeated more than a threshold.

    Returns:
        bool: True if the token is repetitive, False otherwise.
    """

    for char in set(token):
        if token.count(char) > threshold:
            return True
    return False

def clean_text(text):
    """
    Clean the text by expanding contractions, removing HTML tags, and special characters.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    text = contractions.fix(text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', ' ', text)
    text = re.sub(r'([^\s\w]|_)+', lambda match: ' ' + match.group(0) + ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower()

def tokenize_and_stem(text):
    """
    Tokenize, remove stopwords, and stemming the text.

    Returns:
        list: A list of processed tokens.
    """

    text = replace_numbers(text)
    tokens = word_tokenize(text)
    return ' '.join([stemmer.stem(word) for word in tokens
                     if word.isalpha() and 
                     word.lower() not in stop_words and len(word) > 3 and 
                     not is_repetitive(word)])

def build_vocab(tokenized_texts, min_freq=2, max_vocab_size=25000):
    """
    Build a vocabulary from tokenized texts. 

    Parameters:
        tokenized_texts (list of list of str): Tokenized texts.
        min_freq (int): Minimum frequency for words to be included in the vocabulary.
        max_vocab_size (int): Maximum size of the vocabulary.

    Returns:
        dict: A dictionary mapping from words to their indices.
    """

    if not isinstance(tokenized_texts, list) or not all(isinstance(tokens, list) for tokens in tokenized_texts):
        raise TypeError("Invalid input type for build_vocab. Expected a list of tokenized texts.")    
    
    word_freq = Counter(word for tokens in tokenized_texts for word in tokens)
    most_common_words = word_freq.most_common(max_vocab_size)
    vocab = {word for word, freq in most_common_words if freq >= min_freq}
    vocab.add('<UNK>')  # Token for unknown words
    word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
    return word_to_idx

def text_to_sequence(tokenized_text, word_to_idx):
    """
    Convert tokenized text to a sequence of integers.

    Parameters:
        tokenized_text (list of str): A tokenized text.
        word_to_idx (dict): A vocabulary mapping words to indices.

    Returns:
        list: A list of integers representing the text.
    """

    if not isinstance(tokenized_text, list):
        raise TypeError("tokenized_text must be a list of tokens.")

    return [word_to_idx.get(word, word_to_idx['<UNK>']) for word in tokenized_text]

def pad_sequences(sequences, max_len):
    """
    Pad sequences to a uniform length.

    Parameters:
        sequences (list of list of int): A list of sequences.
        max_len (int): The maximum length of a sequence.

    Returns:
        list of list of int: Padded sequences.
    """

    logging.debug("Padding sequences.")
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded_sequences.append(seq)
    return padded_sequences

def prepare_data_for_analysis(dataframe, text_column, sentiment_column):
    """
    Preprocess DataFrame for sentiment analysis with TfidfVectorizer.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the dataset.
        text_column (str): The name of the column containing text.
        sentiment_column (str): The name of the column containing sentiment labels.

    Returns:
        tuple: A tuple containing texts and labels.
    """

    required_columns = {text_column, sentiment_column}
    if not required_columns.issubset(dataframe.columns):
        missing_columns = required_columns - set(dataframe.columns)
        raise KeyError(f"Missing required columns: {missing_columns}")

    try:
        logging.info("Starting preprocessing of data.")

        logging.debug("Cleaning text.")
        dataframe['cleaned_text'] = dataframe[text_column].apply(clean_text)
        dataframe['processed_text'] = dataframe['cleaned_text'].apply(tokenize_and_stem)

        # Process sentiment labels
        # Assuming sentiment labels are 'positive' and 'negative', and converting them to binary values
        labels = dataframe[sentiment_column].apply(lambda x: 1 if x == 'positive' else 0).values

        logging.info("Preprocessing complete.")
        return dataframe['processed_text'], labels
    except Exception as e:
        logging.error(f"Error during preprocessing data: {e}")
        return [], []
