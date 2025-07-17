# """
# Request Model using Pydantic for deployment schemas.
# This code defines a Pydantic model for validating deployment request data.
# """

# # import necessary libraries
# from pydantic import BaseModel

# class TweetRequest(BaseModel):
#     tweet: str

"""
Pydantic schemas for API request/response models
"""

from pydantic import BaseModel, ConfigDict

class TweetRequest(BaseModel):
    tweet: str
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tweet": "I love this new iPhone! It's amazing!"
            }
        }
    )