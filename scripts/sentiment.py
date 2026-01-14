"""
This script defines a SentimentModel class that loads a pre-trained sentiment analysis model
and provides a predict function to analyze the sentiment of given text reviews.
"""

from transformers import pipeline
import torch
import os

class SentimentModel:
    def __init__(self, model_path="sentiment_model"):
        """
        Initialize the sentiment analysis model.
        """
        self.model_path = model_path
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} does not exist.")
        
        print(f"Loading sentiment analysis pipeline from path: {self.model_path}")
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "text-classification",
            model=self.model_path,
            device=device,
            tokenizer=self.model_path,
            truncation=True,
            max_length=512
        )

    
    def predict(self, text):
        """
        Returns the sentiment label and confidence score for the given review.
        """
        # in case the review is too long, truncate it
        text = text[:2000]
        result = self.pipeline(text)[0]
        return result['label'], result['score']