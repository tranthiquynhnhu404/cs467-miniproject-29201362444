import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "this", "to",
    "was", "we", "with", "you", "your"
}

POSITIVE_WORDS = {
    "beautiful", "excellent", "exceeded", "good", "great", "love", "loved",
    "perfect", "personal", "pleasant", "quality", "stylish", "thoughtful",
    "useful", "wonderful", "worth", "charming", "easy"
}
NEGATIVE_WORDS = {
    "average", "disappointing", "poor", "weak", "bad", "not", "never",
    "impressed", "overpriced", "smaller", "durability", "best"
}


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text).lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    tokens = [token for token in tokens if token not in STOP_WORDS]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return clean_text(text).split()


class GiftNLPPipeline:
    def __init__(self, ngram_range=(1, 2), max_features=500):
        self.vectorizer = TfidfVectorizer(
            preprocessor=clean_text,
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )
        self.tfidf = None
        self.item_texts = None
        self.item_ids = None

    def fit_items(self, items: pd.DataFrame) -> None:
        parts = pd.DataFrame({
            "item_id": items["item_id"].astype(str),
            "text": (
                items["gift_name"].fillna("").astype(str) + " " +
                items["occasion_tags"].fillna("").astype(str) + " " +
                items["description"].fillna("").astype(str) + " " +
                items["recipient_tags"].fillna("").astype(str)
            ),
        })
        self.item_ids = parts["item_id"].tolist()
        self.item_texts = parts["text"].tolist()
        self.tfidf = self.vectorizer.fit_transform(self.item_texts)

    def transform_query(self, text: str):
        return self.vectorizer.transform([text])

    def transform_items(self, texts: Iterable[str]):
        return self.vectorizer.transform(list(texts))

    def top_keywords(self, vector, text: str = "", top_n: int = 5) -> list[str]:
        feature_names = np.asarray(self.vectorizer.get_feature_names_out())
        values = vector.toarray().ravel()
        indices = values.argsort()[::-1][:top_n]
        keywords = [feature_names[i] for i in indices if values[i] > 0]
        return keywords[:top_n]

    def sentiment_score(self, text: str) -> float:
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        positive = sum(token in POSITIVE_WORDS for token in tokens)
        negative = sum(token in NEGATIVE_WORDS for token in tokens)
        return float(np.clip((positive - negative) / len(tokens), -1.0, 1.0))

    def item_sentiments(self, ratings: pd.DataFrame) -> pd.Series:
        scores = ratings["buyer_review"].fillna("").map(self.sentiment_score)
        return pd.DataFrame({
            "item_id": ratings["item_id"],
            "sentiment": scores,
        }).groupby("item_id")["sentiment"].mean()

