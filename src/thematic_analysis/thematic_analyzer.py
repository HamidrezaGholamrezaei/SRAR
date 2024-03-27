"""
Thematic Analyzer Module for Smart Review Analyzer and Recommender (SRAR)

This module is responsible for conducting thematic analysis on customer reviews using
Non-negative Matrix Factorization (NMF) for topic modeling. It aims to provide insights
into prevalent themes within the data, facilitating a deeper understanding of customer sentiments.
"""


import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from gensim.models import CoherenceModel
from gensim import corpora
from gensim.utils import simple_preprocess


class ThematicAnalyzer:
    """
    A class for performing thematic analysis using NMF and evaluating topic coherence.
    
    Attributes:
        config (argparse.Namespace): Configuration parameters including number of topics and features.
        documents (list): Collection of documents to analyze.

    Methods:
        execute_model_workflow(): Conducts the complete thematic analysis workflow.
        fit_transform(): Applies TF-IDF and fits the NMF model on the documents.
        display_topics(feature_names: list): Logs the top words for each identified topic.
        prepare_corpus(): Prepares the corpus for coherence calculation.
        compute_topic_coherence(H: numpy.ndarray): Calculates and logs the coherence of the topics.
        visualize_topics(H: numpy.ndarray, feature_names: list): Generates and saves a heatmap of topic-word associations.
    """

    def __init__(self, config, documents):
        self.config = config
        self.documents = documents
        self.n_topics = config.n_topics
        self.n_features = config.n_features
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=self.n_features, stop_words='english')
        try:
            self.nmf_model = NMF(n_components=self.n_topics, random_state=config.seed, alpha=.1, l1_ratio=.5)
        except Exception as e:
            logging.error(f"Failed to initialize NMF model: {e}")
            raise
        logging.info("ThematicAnalyzer initialized successfully.")

    def execute_model_workflow(self):
        logging.info("Starting thematic analysis...")
        W, H, tfidf_matrix = self.fit_transform()
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.display_topics(feature_names)
        self.visualize_topics(H, feature_names)
        self.compute_topic_coherence(H)
        logging.info("Thematic analysis completed.")

    def fit_transform(self):
        try:
            tfidf = self.tfidf_vectorizer.fit_transform(self.documents)
            W = self.nmf_model.fit_transform(tfidf)
            H = self.nmf_model.components_
            return W, H, tfidf
        except Exception as e:
            logging.error(f"Failed to fit_transform documents: {e}")
            raise
    
    def display_topics(self, feature_names):
        logging.info("Top words per topic:")
        for topic_idx, topic in enumerate(self.nmf_model.components_):
            message = f"Topic #{topic_idx}: " + " ".join([feature_names[i] for i in topic.argsort()[:-self.config.no_top_words - 1:-1]])
            logging.info(message)
    
    def prepare_corpus(self):
        try:
            tokenized_documents = [simple_preprocess(doc) for doc in self.documents]
            dictionary = corpora.Dictionary(tokenized_documents)
            corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]
            return corpus, dictionary, tokenized_documents
        except Exception as e:
            logging.error(f"Failed to prepare corpus: {e}")
            raise
    
    def compute_topic_coherence(self, H):
        corpus, dictionary, tokenized_documents = self.prepare_corpus()
        top_words_per_topic = []
        for topic_idx, topic in enumerate(H):
            top_indices = topic.argsort()[-self.config.no_top_words:]
            top_words = [self.tfidf_vectorizer.get_feature_names_out()[i] for i in top_indices]
            top_words_per_topic.append(top_words)
        
        coherence_model = CoherenceModel(topics=top_words_per_topic, texts=tokenized_documents, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        logging.info(f"Average Topic Coherence: {coherence_score:.4f}")

    def visualize_topics(self, H, feature_names):
        H_normalized = normalize(H, norm='l1', axis=1)
        topic_word_df = pd.DataFrame(H_normalized, columns=feature_names)
        
        plt.figure(figsize=(15, 10))
        heatmap = sns.heatmap(topic_word_df, cmap=sns.diverging_palette(240, 10, n=9), annot=True, fmt=".2f")
        heatmap.set_title('Topic-Word Associations', fontdict={'fontsize':18}, pad=16)
        heatmap.set_xlabel('Words', fontsize=14)
        heatmap.set_ylabel('Topics', fontsize=14)
        
        # for i, topic in enumerate(H_normalized):
        #     heatmap.annotate(f"Key: {feature_names[np.argmax(topic)]}", 
        #                     xy=(0, i), 
        #                     xytext=(-3.7, i), 
        #                     textcoords='offset points',
        #                     va='center',
        #                     ha='right',
        #                     fontsize=12,
        #                     color='blue')
        
        figure_path = os.path.join(self.config.evaluation_directory, f'topic_word_heatmap_{self.config.dataset}.png')
        plt.savefig(figure_path, bbox_inches='tight')
        plt.close()
