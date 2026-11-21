from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from nlp_pipeline import GiftNLPPipeline


class GiftRecommender:
    def __init__(self, items: pd.DataFrame, ratings: pd.DataFrame, nlp: GiftNLPPipeline):
        self.items = items.copy()
        self.ratings = ratings.copy()
        self.nlp = nlp
        self.nlp.fit_items(self.items)
        self.id_to_row = {
            item_id: index
            for index, item_id in enumerate(self.items["item_id"].astype(str).tolist())
        }
        self.popularity = self._compute_popularity(min_count=2)
        self.item_sentiment = self.nlp.item_sentiments(self.ratings)

    def _compute_popularity(self, min_count: int = 2) -> pd.Series:
        stats = self.ratings.groupby("item_id")["rating"].agg(["mean", "count"])
        stats = stats[stats["count"] >= min_count].copy()
        stats["score"] = stats["mean"] * np.log1p(stats["count"])
        return stats["score"].fillna(0.0)

    def popularity_recommend(self, user_id: str | None = None, top_n: int = 5) -> list[dict]:
        seen = set()
        if user_id is not None:
            seen = set(self.ratings.loc[
                self.ratings["user_id"].astype(str) == str(user_id), "item_id"
            ].astype(str))
        return self._rank(self.popularity, seen, top_n)

    def content_recommend_user(self, user_id: str, top_n: int = 5) -> list[dict]:
        seen_ratings = self.ratings[
            self.ratings["user_id"].astype(str) == str(user_id)
        ]
        seen = set(seen_ratings["item_id"].astype(str))
        if seen_ratings.empty:
            return self.popularity_recommend(user_id=user_id, top_n=top_n)

        weights = []
        vectors = []
        for row in seen_ratings.itertuples():
            item_id = str(row.item_id)
            if item_id in self.id_to_row:
                vectors.append(self.nlp.tfidf[self.id_to_row[item_id]])
                weights.append(float(row.rating) - 2.5)
        if not vectors:
            return self.popularity_recommend(user_id=user_id, top_n=top_n)

        profile = sum(weight * vector for weight, vector in zip(weights, vectors))
        scores = cosine_similarity(profile, self.nlp.tfidf).ravel()
        return self._rank_scores(scores, seen, top_n)

    def content_recommend_query(self, query: str, top_n: int = 5) -> list[dict]:
        query_vector = self.nlp.transform_query(query)
        scores = cosine_similarity(query_vector, self.nlp.tfidf).ravel()
        return self._rank_scores(scores, set(), top_n)

    def similar_items(self, item_id: str, top_n: int = 5) -> list[dict]:
        item_id = str(item_id)
        if item_id not in self.id_to_row:
            return []
        row = self.nlp.tfidf[self.id_to_row[item_id]]
        scores = cosine_similarity(row, self.nlp.tfidf).ravel()
        return self._rank_scores(scores, {item_id}, top_n)

    def hybrid_recommend(self, user_id: str | None = None, query: str | None = None,
                         top_n: int = 5) -> list[dict]:
        if query:
            content = cosine_similarity(
                self.nlp.transform_query(query), self.nlp.tfidf
            ).ravel()
            seen = set()
        elif user_id:
            user_ratings = self.ratings[
                self.ratings["user_id"].astype(str) == str(user_id)
            ]
            seen = set(user_ratings["item_id"].astype(str))
            if user_ratings.empty:
                content = np.zeros(len(self.items))
            else:
                vectors, weights = [], []
                for row in user_ratings.itertuples():
                    item_id = str(row.item_id)
                    if item_id in self.id_to_row:
                        vectors.append(self.nlp.tfidf[self.id_to_row[item_id]])
                        weights.append(float(row.rating) - 2.5)
                profile = sum(weight * vector for weight, vector in zip(weights, vectors))
                content = cosine_similarity(profile, self.nlp.tfidf).ravel()
        else:
            content = np.zeros(len(self.items))
            seen = set()

        item_ids = self.items["item_id"].astype(str).tolist()
        popularity_values = np.array([
            self.popularity.get(item_id, 0.0) for item_id in item_ids
        ])
        popularity_norm = self._minmax(popularity_values)
        sentiment_values = np.array([
            (self.item_sentiment.get(item_id, 0.0) + 1.0) / 2.0
            for item_id in item_ids
        ])
        content_norm = self._minmax(content)

        hybrid = 0.45 * content_norm + 0.25 * sentiment_values + 0.30 * popularity_norm
        return self._rank_scores(hybrid, seen, top_n)

    def add_cold_start_item(self, item_id: str, gift_name: str, description: str,
                            occasion_tags: str = "", recipient_tags: str = "") -> str:
        row = pd.DataFrame([{
            "item_id": item_id,
            "gift_name": gift_name,
            "price_range": "seasonal",
            "occasion_tags": occasion_tags,
            "description": description,
            "recipient_tags": recipient_tags,
        }])
        self.items = pd.concat([self.items, row], ignore_index=True)
        self.nlp.fit_items(self.items)
        self.id_to_row = {
            item_id: index
            for index, item_id in enumerate(self.items["item_id"].astype(str).tolist())
        }
        return item_id

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        if len(values) == 0 or values.max() == values.min():
            return np.zeros_like(values, dtype=float)
        return (values - values.min()) / (values.max() - values.min())

    def _rank_scores(self, scores: np.ndarray, seen: set[str],
                     top_n: int) -> list[dict]:
        ranked = []
        for index, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True):
            item = self.items.iloc[index]
            item_id = str(item["item_id"])
            if item_id in seen:
                continue
            query_vector = self.nlp.tfidf[index:index + 1]
            ranked.append({
                "item_id": item_id,
                "gift_name": item["gift_name"],
                "price_range": item["price_range"],
                "score": float(score),
                "keywords": self.nlp.top_keywords(query_vector),
            })
            if len(ranked) == top_n:
                break
        return ranked

    def _rank(self, values: pd.Series, seen: set[str], top_n: int) -> list[dict]:
        ranked = []
        for item_id, score in values.sort_values(ascending=False).items():
            item_id = str(item_id)
            if item_id in seen or item_id not in self.id_to_row:
                continue
            item = self.items.iloc[self.id_to_row[item_id]]
            row = self.nlp.tfidf[self.id_to_row[item_id]:self.id_to_row[item_id] + 1]
            ranked.append({
                "item_id": item_id,
                "gift_name": item["gift_name"],
                "price_range": item["price_range"],
                "score": float(score),
                "keywords": self.nlp.top_keywords(row),
            })
            if len(ranked) == top_n:
                break
        return ranked

