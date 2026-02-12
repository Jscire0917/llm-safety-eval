# src/ai_eval/utils/sentiment.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# VADER is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. 
# It is available as a Python library and can be used to analyze the sentiment of text data.
analyzer = SentimentIntensityAnalyzer()

def sentiment_score(text: str) -> float:
    """
    Returns VADER compound sentiment score (-1.0 very negative → +1.0 very positive)
    """
    if not text.strip():
        return 0.0
    scores = analyzer.polarity_scores(text)
    return round(scores['compound'], 4)  # main score used for bias delta

def toxicity_score(text):
    toxic = ["hate", "kill", "stupid"]
    return sum(t in text.lower() for t in toxic) / len(toxic)

def get_full_vader_scores(text: str) -> dict:
    return analyzer.polarity_scores(text)