import pytest
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from tweet_preprocessor import TweetPreprocessor

@pytest.fixture
def processor():
    return TweetPreprocessor()

@pytest.fixture  
def test_processor():
    """TweetPreprocessor with test-friendly parameters for small datasets"""
    return TweetPreprocessor(
        tfidf_min_df=1,  # Allow terms that appear in at least 1 document
        tfidf_max_df=1.0,  # Allow terms that appear in up to 100% of documents
        tfidf_max_features=100,  # Reduce for small test datasets
        tfidf_ngram_range=(1, 1)  # Use unigrams only for simpler testing
    )

def test_expand_contractions(processor):
    text = "I can't believe it's already done"
    expected = "I cannot believe it is already done"
    result = processor.expand_contractions_text(text)
    
    print(f"\nContraction Expansion Test:")
    print(f"   Input:    '{text}'")
    print(f"   Output:   '{result}'")
    print(f"   Expected: '{expected}'")
    
    assert result == expected

def test_remove_repeated_characters(processor):
    text = "sooooo happpppyyyy!!!"
    expected = "soo happyy!!"
    result = processor.remove_repeated_characters(text)
    
    print(f"\nRepeated Character Removal Test:")
    print(f"   Input:    '{text}'")
    print(f"   Output:   '{result}'")
    print(f"   Expected: '{expected}'")
    
    assert result == expected

def test_clean_text(processor):
    text = "@user I'm sooooo happyyyy today!!! #excited http://example.com"
    cleaned = processor.clean_text(text)
    
    print(f"\n🧹 TEXT CLEANING TEST:")
    print(f"   Input:   '{text}'")
    print(f"   Output:  '{cleaned}'")
    print(f"   Removed: URLs={'http' not in cleaned}, Mentions={'@' not in cleaned}, Hashtags={'#' not in cleaned}")
    print(f"   Contains: happy={'happy' in cleaned or 'happyy' in cleaned}, excited={'excited' in cleaned}")
    
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
    assert "happy" in cleaned or "happyy" in cleaned
    assert "excited" in cleaned

def test_preprocess_text(processor):
    text = "I'm extremelyyyyyyyyyyyy excited about this!!"
    cleaned = processor.clean_text(text)
    preprocessed = processor.preprocess_text(cleaned)
    tokens = preprocessed.split()
    
    print(f"\n📝 TEXT PREPROCESSING TEST:")
    print(f"   Original:     '{text}'")
    print(f"   Cleaned:      '{cleaned}'")
    print(f"   Preprocessed: '{preprocessed}'")
    print(f"   Tokens:       {tokens}")
    print(f"   Min length check: All tokens >= {processor.min_length}: {all(len(t) >= processor.min_length for t in tokens)}")
    
    assert "excited" in tokens
    assert all(len(t) >= processor.min_length for t in tokens)

def test_fit_transform_shape(test_processor):
    tweets = [
        "I'm loving the new feature!! #awesome",
        "I can't believe this happened... @someone",
        "So bored of this... what's next?",
        "Visit our website: https://test.com"
    ]
    X_transformed = test_processor.fit_transform(tweets)
    
    print(f"\n🔧 FIT_TRANSFORM TEST:")
    print(f"   Input tweets: {len(tweets)}")
    for i, tweet in enumerate(tweets):
        print(f"     {i+1}. '{tweet}'")
    print(f"   Output shape: {X_transformed.shape}")
    print(f"   Matrix type: {type(X_transformed)}")
    print(f"   Has vectorizer: {hasattr(test_processor, 'vectorizer')}")
    print(f"   Vectorizer type: {type(test_processor.vectorizer).__name__}")
    
    assert X_transformed.shape[0] == 4
    assert hasattr(test_processor, 'vectorizer')
    assert test_processor.vectorizer is not None

def test_get_feature_names(test_processor):
    tweets = ["Happy coding!", "Sad times."]
    test_processor.fit(tweets)
    feature_names = test_processor.get_feature_names_out()
    
    print(f"\n📊 FEATURE NAMES TEST:")
    print(f"   Input tweets: {tweets}")
    print(f"   Feature names: {feature_names[:10]}...")  # Show first 10
    print(f"   Total features: {len(feature_names)}")
    print(f"   Feature type: {type(feature_names)}")
    
    assert isinstance(feature_names, np.ndarray)
    assert len(feature_names) > 0

def test_get_vocabulary(test_processor):
    tweets = ["Happy coding!", "Sad times."]
    test_processor.fit(tweets)
    vocab = test_processor.get_vocabulary()
    
    print(f"\n📚 VOCABULARY TEST:")
    print(f"   Input tweets: {tweets}")
    print(f"   Vocabulary size: {len(vocab)}")
    print(f"   Sample vocab items: {dict(list(vocab.items())[:5])}")
    print(f"   Contains 'happy': {'happy' in vocab}")
    print(f"   Contains 'coding': {'coding' in vocab}")
    
    assert isinstance(vocab, dict)
    assert "happy" in vocab or "coding" in vocab

# def test_pipeline_integration(test_processor):
#     """Test that TweetPreprocessor works in a scikit-learn pipeline"""
#     tweets = [
#         "I'm so happy about this!",
#         "This is the worst day ever.",
#         "Absolutely loved it!",
#         "I hate everything."
#     ]
#     y = [1, 0, 1, 0]  # Sentiment labels

#     pipeline = Pipeline([
#         ('preprocessor', test_processor),
#         ('classifier', LogisticRegression(random_state=42))
#     ])

#     pipeline.fit(tweets, y)
#     test_tweet = ["I'm not happy with this at all"]
#     preds = pipeline.predict(test_tweet)
    
#     print(f"\n🔗 PIPELINE INTEGRATION TEST:")
#     print(f"   Training tweets: {len(tweets)}")
#     print(f"   Training labels: {y}")
#     print(f"   Test tweet: {test_tweet[0]}")
#     print(f"   Prediction: {preds[0]} (0=negative, 1=positive)")
#     print(f"   Pipeline steps: {[step[0] for step in pipeline.steps]}")
    
#     assert preds[0] in [0, 1]

if __name__ == "__main__":
    # Run tests with verbose output when executed directly
    print("🚀 Running TweetPreprocessor Tests with Visual Output")
    print("=" * 60)
    
    processor = TweetPreprocessor()
    test_proc = TweetPreprocessor(
        tfidf_min_df=1,
        tfidf_max_df=1.0,
        tfidf_max_features=100,
        tfidf_ngram_range=(1, 1)
    )
    
    test_expand_contractions(processor)
    test_remove_repeated_characters(processor)
    test_clean_text(processor)
    test_preprocess_text(processor)
    test_fit_transform_shape(test_proc)
    test_get_feature_names(test_proc)
    test_get_vocabulary(test_proc)
    # test_pipeline_integration(test_proc)
    
    print(f"\n🎉 All tests completed successfully!")
