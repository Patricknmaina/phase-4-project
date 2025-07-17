"""
Tweet Preprocessor Class for Sentiment Analysis

This module contains the TweetPreprocessor class that handles data cleaning,
text preprocessing, and feature extraction for Twitter sentiment analysis.
"""

import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class TweetPreprocessor(BaseEstimator, TransformerMixin):
    """
    A preprocessing class for Twitter Sentiment Analysis.

    This class handles:
    - Data cleaning (removing URLs, mentions, hashtags, special characters)
    - Text preprocessing (tokenization, lemmatization, stop words removal)
    - Feature extraction using TF-IDF or Count Vectorization

    This class can be implemented directly in a scikit-learn pipeline
    """

    def __init__(self,
                 remove_urls=True,
                 remove_mentions=True,
                 remove_hashtags=True,
                 remove_stopwords=True,
                 lemmatize=True,
                 lowercase=True,
                 min_length=2,
                 expand_contractions=True,  # New feature
                 remove_repeated_chars=True,  # New feature
                 tfidf_max_features=5000,
                 tfidf_ngram_range=(1, 2),
                 tfidf_min_df=1,  # Changed from 1 to prevent errors
                 tfidf_max_df=1.0,  # Changed from 0.95 to 1.0 for safer handling
                 use_tfidf=True):
        """
        Initialize the TweetPreprocessor.

        Parameters:
        -----------
        remove_urls : bool, default=True
            Whether to remove URLs from tweets
        remove_mentions : bool, default=True
            Whether to remove @mentions from tweets
        remove_hashtags : bool, default=True
            Whether to remove #hashtags from tweets (often contain sentiment info)
        remove_stopwords : bool, default=True
            Whether to remove stop words
        lemmatize : bool, default=True
            Whether to reduce words to their base form
        lowercase : bool, default=True
            Whether to convert text to lowercase
        min_length : int, default=2
            Minimum word length to keep
        expand_contractions : bool, default=True
            Whether to expand contractions (e.g., "can't" -> "cannot")
        remove_repeated_chars : bool, default=True
            Whether to reduce repeated characters (e.g., "goooood" -> "good")
        tfidf_max_features : int, default=15000
            Maximum number of features for TF-IDF
        tfidf_ngram_range : tuple, default=(1, 3)
            N-gram range for TF-IDF
        tfidf_min_df : int, default=1
            Minimum document frequency for terms (default 1)
        tfidf_max_df : float, default=1.0
            Maximum document frequency cutoff (default 1.0)
        use_tfidf : bool, default=True
            Whether to use TF-IDF (True) or Count Vectorizer (False)
        """
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.min_length = min_length
        self.expand_contractions = expand_contractions
        self.remove_repeated_chars = remove_repeated_chars
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.use_tfidf = use_tfidf

        # initialize the components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None

        # Contraction mapping
        self.contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "it's": "it is",
            "that's": "that is", "what's": "what is",
            "there's": "there is", "here's": "here is"
        }

    def _extract_text_from_input(self, X):
        """
        Extract text data from various input formats (DataFrame, Series, list, etc.)
        """
        if hasattr(X, 'values'):
            # Handle pandas DataFrame or Series
            if hasattr(X, 'columns'):
                # DataFrame - look for 'tweet' column or use first column
                if 'tweet' in X.columns:
                    return X['tweet'].values
                else:
                    return X.iloc[:, 0].values
            else:
                # Series
                return X.values
        elif isinstance(X, (list, tuple)):
            # Handle list or tuple
            return X
        else:
            # Handle numpy array or other array-like
            return X

    def expand_contractions_text(self, text):
        """Expand contractions in text"""
        if not self.expand_contractions:
            return text
        
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def remove_repeated_characters(self, text):
        """Remove repeated characters (e.g., 'goooood' -> 'good')"""
        if not self.remove_repeated_chars:
            return text
        
        # Replace 3+ repeated characters with 2
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    # data cleaning function
    def clean_text(self, text):
        """
        Cleans individual tweet text

        Parameters:
        -----------
        text: str
            The tweet text to clean
        """

        # return empty string if text is NaN
        if pd.isna(text):
            return ''
        
        # convert to string if not already
        text = str(text)

        # Expand contractions first
        text = self.expand_contractions_text(text)
        
        # Remove repeated characters
        text = self.remove_repeated_characters(text)

        # remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # remove mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)

        # remove hashtags (but keep the text)
        if self.remove_hashtags:
            text = re.sub(r'#', '', text)

        # remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s!?]', '', text)

        # Convert multiple exclamation/question marks to single
        text = re.sub(r'!+', '!', text)
        text = re.sub(r'\?+', '?', text)

        # remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # text preprocessing function
    def preprocess_text(self, text):
        """
        Preprocesses the cleaned text (tokenization, lemmatization, stop words removal)
        
        Parameters:
        -----------
        text: str
            The cleaned tweet text to preprocess

        Returns:
        --------
        str
            The preprocessed tweet text
        """

        # return an empty string if not text
        if not text:
            return ''
        
        # convert text to lowercase
        if self.lowercase:
            text = text.lower()

        # tokenize the text
        tokens = word_tokenize(text)

        # remove stop words and short words
        if self.remove_stopwords:
            tokens = [token for token in tokens
                      if token not in self.stop_words and len(token) >= self.min_length]
        else:
            tokens = [token for token in tokens if len(token) >= self.min_length]

        # lemmatize the tokens
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(tokens)

    # function to fit the vectorizer
    def fit(self, X, y=None):
        """
        Fit the preprocessor to the data

        Parameters:
        -----------
        X: array-like
           Input tweets
        y: array-like, optional
           Target labels

        Returns:
        --------
        self
        """

        # Extract text data from input
        text_data = self._extract_text_from_input(X)

        # clean and preprocess the tweets
        processed_texts = []
        for text in text_data:
            cleaned = self.clean_text(text)
            preprocessed = self.preprocess_text(cleaned)
            processed_texts.append(preprocessed)

        # Automatically adjust parameters for small datasets
        n_docs = len(processed_texts)
        min_df = min(self.tfidf_min_df, max(1, n_docs // 100))  # At least 1, at most n_docs/100
        max_df = min(self.tfidf_max_df, 1.0)  # Ensure max_df is never > 1.0
        
        # Ensure min_df doesn't exceed reasonable bounds
        if min_df >= n_docs * max_df:
            min_df = 1
            max_df = 1.0

        # fit the TF-IDF vectorizer
        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=self.tfidf_ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words='english'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=self.tfidf_ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words='english'
            )
        self.vectorizer.fit(processed_texts)

        return self
    
    # function to transform the data
    def transform(self, X):
        """
        Transform the input data using the fitted vectorizer

        Parameters:
        -----------
        X: array-like
           Input tweets

        Returns:
        --------
        scipy.sparse matrix
            TF-IDF transformed features
        """

        # check for fitted vectorizer
        if self.vectorizer is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")

        # Extract text data from input
        text_data = self._extract_text_from_input(X)

        # process all the tweets
        processed_texts = []
        for text in text_data:
            cleaned = self.clean_text(text)
            preprocessed = self.preprocess_text(cleaned)
            processed_texts.append(preprocessed)

        # transform using the fitted vectorizer
        return self.vectorizer.transform(processed_texts)
    
    # fit_transform function
    def fit_transform(self, X, y=None):
        """
        Fit and transform the input data

        Parameters:
        -----------
        X: array-like
           Input tweets
        y: array-like, optional
            Target labels

        Returns:
        --------
        scipy.sparse matrix
            TF-IDF transformed features
        """
        return self.fit(X, y).transform(X)
    
    # function to get feature names
    def get_feature_names_out(self, input_features=None):
        """
        Get the feature names of the transformed output
        
        Returns:
        --------
        array
            Feature names from the fitted vectorizer
        """
        if self.vectorizer is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first")
        
        return self.vectorizer.get_feature_names_out()
    
    # function to get the vocabulary
    def get_vocabulary(self):
        """
        Get the vocabulary dictionary.

        Returns:
        --------
        dict
            Vocabulary mapping from words to feature indices
        """

        if self.vectorizer is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first")
        
        return self.vectorizer.vocabulary_
