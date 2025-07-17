"""
Streamlit Web Application for classifying tweets as positive, negative, or neutral sentiments
"""

# import the necessary libraries
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')  
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Define TweetPreprocessor class directly in this file before loading the model
class TweetPreprocessor(BaseEstimator, TransformerMixin):
    """
    A preprocessing class for Twitter Sentiment Analysis.
    """

    def __init__(self,
                 remove_urls=True,
                 remove_mentions=True,
                 remove_hashtags=True,
                 remove_stopwords=True,
                 lemmatize=True,
                 lowercase=True,
                 min_length=2,
                 expand_contractions=True,
                 remove_repeated_chars=True,
                 tfidf_max_features=5000,
                 tfidf_ngram_range=(1, 2),
                 tfidf_min_df=1,
                 tfidf_max_df=1.0,
                 use_tfidf=True):
        
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
        """Extract text data from various input formats"""
        if hasattr(X, 'values'):
            if hasattr(X, 'columns'):
                if 'tweet' in X.columns:
                    return X['tweet'].values
                else:
                    return X.iloc[:, 0].values
            else:
                return X.values
        elif isinstance(X, (list, tuple)):
            return X
        else:
            return X

    def expand_contractions_text(self, text):
        """Expand contractions in text"""
        if not self.expand_contractions:
            return text
        
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def remove_repeated_characters(self, text):
        """Remove repeated characters"""
        if not self.remove_repeated_chars:
            return text
        
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    def clean_text(self, text):
        """Cleans individual tweet text"""
        if pd.isna(text):
            return ''
        
        text = str(text)
        text = self.expand_contractions_text(text)
        text = self.remove_repeated_characters(text)

        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)

        if self.remove_hashtags:
            text = re.sub(r'#', '', text)

        text = re.sub(r'[^a-zA-Z\s!?]', '', text)
        text = re.sub(r'!+', '!', text)
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess_text(self, text):
        """Preprocesses the cleaned text"""
        if not text:
            return ''
        
        if self.lowercase:
            text = text.lower()

        tokens = word_tokenize(text)

        if self.remove_stopwords:
            tokens = [token for token in tokens
                      if token not in self.stop_words and len(token) >= self.min_length]
        else:
            tokens = [token for token in tokens if len(token) >= self.min_length]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(tokens)

    def fit(self, X, y=None):
        """Fit the preprocessor to the data"""
        text_data = self._extract_text_from_input(X)

        processed_texts = []
        for text in text_data:
            cleaned = self.clean_text(text)
            preprocessed = self.preprocess_text(cleaned)
            processed_texts.append(preprocessed)

        n_docs = len(processed_texts)
        min_df = min(self.tfidf_min_df, max(1, n_docs // 100))
        max_df = min(self.tfidf_max_df, 1.0)
        
        if min_df >= n_docs * max_df:
            min_df = 1
            max_df = 1.0

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
    
    def transform(self, X):
        """Transform the input data using the fitted vectorizer"""
        if self.vectorizer is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")

        text_data = self._extract_text_from_input(X)

        processed_texts = []
        for text in text_data:
            cleaned = self.clean_text(text)
            preprocessed = self.preprocess_text(cleaned)
            processed_texts.append(preprocessed)

        return self.vectorizer.transform(processed_texts)
    
    def fit_transform(self, X, y=None):
        """Fit and transform the input data"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Get the feature names of the transformed output"""
        if self.vectorizer is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first")
        
        return self.vectorizer.get_feature_names_out()
    
    def get_vocabulary(self):
        """Get the vocabulary dictionary"""
        if self.vectorizer is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first")
        
        return self.vectorizer.vocabulary_

# Ensure TweetPreprocessor is available in the global context for joblib
sys.modules[__name__].TweetPreprocessor = TweetPreprocessor
import __main__
__main__.TweetPreprocessor = TweetPreprocessor

# load the saved model pipeline
loaded_model = joblib.load('model/multi_nlp_model.pkl')

# Class index to label mapping
label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
emoji_map = {"Positive": "😃", "Negative": "😠", "Neutral": "😐"}

# --- Streamlit UI ---
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")
st.title("💬 Twitter Sentiment Analyzer")
st.markdown("This module analyzes the sentiment of a tweet as **Positive**, **Negative**, or **Neutral** using a Multi-Class Machine Learning model. It is ideal for tech companies such as Apple and Google, which highly value customer feedback and sentiment")

# --- User Input ---
user_input = st.text_area("📥 Enter your tweet below", height=150, max_chars=280)

# --- Prediction Logic ---
if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a tweet to analyze.")
    else:
        # Predict sentiment
        prediction = loaded_model.predict([user_input])[0]
        predicted_label = label_map.get(prediction, "Unknown")
        emoji = emoji_map.get(predicted_label, "")

        # Predict probabilities
        probs = loaded_model.predict_proba([user_input])[0]
        confidence = np.max(probs)

        # --- Display Results ---
        st.markdown(f"### 🎯 Predicted Sentiment: **{predicted_label}** {emoji}")
        st.markdown(f"**Confidence:** {confidence:.2%}")

        # --- Probability Bar Chart ---
        proba_df = pd.DataFrame({
            "Sentiment": [label_map[i] for i in range(len(probs))],
            "Probability": probs
        }).sort_values("Probability", ascending=False)

        st.markdown("#### 📊 Prediction Probabilities")
        # st.bar_chart(proba_df.set_index("Sentiment"))

        # Define color map for sentiment
        color_map = {
            "Positive": "#2ecc71",  # green
            "Negative": "#e74c3c",  # red
            "Neutral": "#f1c40f"    # yellow
        }

        # Create Plotly bar chart
        fig = px.bar(
            proba_df,
            x="Sentiment",
            y="Probability",
            color="Sentiment",
            color_discrete_map=color_map,
            text=proba_df["Probability"].apply(lambda x: f"{x:.2%}"),
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(
            yaxis=dict(title="Probability", range=[0, 1]),
            xaxis=dict(title="Sentiment"),
            showlegend=False,
            title="Prediction Probabilities by Sentiment",
            plot_bgcolor="#ffffff",
        )

        # Display it
        st.plotly_chart(fig, use_container_width=True)


        # Optional: Detailed Table
        with st.expander("🔎 Show detailed probabilities"):
            st.dataframe(proba_df.set_index("Sentiment"))

# --- Footer ---
st.markdown("---")
st.markdown("Built by [Patrick Maina](https://github.com/Patricknmaina) | Powered by Streamlit & Scikit-learn")
