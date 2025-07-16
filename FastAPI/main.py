# """
# Main entry point for the deployment service.
# Implements the FastAPI application and defines the API endpoints.
# """

# # import necessary libraries
# from fastapi import FastAPI
# from schemas import TweetRequest
# from model import predict_sentiment
# from tweet_preprocessor import TweetPreprocessor

# app = FastAPI()

# @app.post("/predict")
# def predict(tweet_request: TweetRequest) -> str:
#     """
#     API endpoint for predicting tweet sentiment.
#     """
#     return predict_sentiment(tweet_request)

"""
Main entry point for the deployment service.
Implements the FastAPI application and defines the API endpoints.
"""

from fastapi import FastAPI, HTTPException
from schemas import TweetRequest
from model import predict_sentiment

app = FastAPI(
    title="Multi-class Twitter Sentiment Analysis API",
    description="An API for predicting tweet sentiments using a multi-class classification model.",
    version="1.0.0"
)

# gets the project root directory
@app.get("/")
def root():
    """Health check endpoint"""
    return {"message": "Twitter Sentiment Analysis API is running!"}

# posts an endpoint for predicting tweet sentiment
@app.post("/predict")
def predict(tweet_request: TweetRequest):
    """
    API endpoint for predicting tweet sentiment.
    
    Args:
        tweet_request: Request body containing the tweet text
        
    Returns:
        Prediction result with sentiment and confidence
    """
    result = predict_sentiment(tweet_request)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"]) # Handle prediction errors
    
    return result
