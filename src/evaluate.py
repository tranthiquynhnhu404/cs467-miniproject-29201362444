import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from nlp_pipeline import GiftNLPPipeline
from recommender import GiftRecommender


def leave_one_out_split(ratings: pd.DataFrame, seed: int = 467):
    random.seed(seed)
    test_indices = []
    train_indices = []
    for _, group in ratings.groupby("user_id"):
        indices = group.index.tolist()
        hidden = random.sample(indices, k=max(4, int(round(len(indices) * 0.20))))
        test_indices.extend(hidden)
        train_indices.extend([i for i in indices if i not in hidden])
    return ratings.loc[train_indices].copy(), ratings.loc[test_indices].copy()


def precision_at_k(recommender, train_ratings, test_ratings, k=5, method="popularity"):
    recommender.ratings = train_ratings.copy()
    recommender.popularity = recommender._compute_popularity(min_count=2)
    recommender.item_sentiment = recommender.nlp.item_sentiments(train_ratings)

    precisions = []
    for user_id, group in test_ratings.groupby("user_id"):
        relevant = set(group.loc[group["rating"] >= 4, "item_id"].astype(str))
        if not relevant:
            continue
        if method == "popularity":
            recs = recommender.popularity_recommend(str(user_id), k)
        elif method == "content":
            recs = recommender.content_recommend_user(str(user_id), k)
        else:
            recs = recommender.hybrid_recommend(str(user_id), top_n=k)
        recommended = {r["item_id"] for r in recs}
        precisions.append(len(relevant & recommended) / k)
    return float(np.mean(precisions)) if precisions else 0.0


def rating_predictions(recommender, test_ratings):
    overall_mean = recommender.ratings["rating"].mean()
    actual, predicted = [], []
    for row in test_ratings.itertuples():
        actual.append(float(row.rating))
        user_mean = recommender.ratings.loc[
            recommender.ratings["user_id"] == row.user_id, "rating"
        ].mean()
        item_mean = recommender.ratings.loc[
            recommender.ratings["item_id"] == row.item_id, "rating"
        ].mean()
        prediction = item_mean if not np.isnan(item_mean) else (
            user_mean if not np.isnan(user_mean) else overall_mean
        )
        predicted.append(float(prediction))
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    return mae, rmse


def plot_rating_distribution(ratings, output_path):
    counts = ratings["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#5B8FF9")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of interactions")
    ax.set_title("Gift Rating Distribution")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric_comparison(metrics, output_path):
    names = list(metrics)
    values = [metrics[name]["precision_at_5"] for name in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=["#91D5FF", "#73D13D", "#FF9C6E"])
    ax.set_ylabel("Precision@5")
    ax.set_ylim(0, max(values + [0.1]) * 1.25)
    ax.set_title("Recommendation Method Comparison")
    ax.grid(axis="y", alpha=.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}",
                ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate(items, ratings, results_dir):
    train, test = leave_one_out_split(ratings)
    nlp = GiftNLPPipeline()
    recommender = GiftRecommender(items, train, nlp)

    metrics = {
        "Popularity": {
            "precision_at_5": precision_at_k(recommender, train, test, 5, "popularity")
        },
        "NLP Content": {
            "precision_at_5": precision_at_k(recommender, train, test, 5, "content")
        },
        "Sentiment Hybrid": {
            "precision_at_5": precision_at_k(recommender, train, test, 5, "hybrid")
        },
    }
    mae, rmse = rating_predictions(recommender, test)
    metrics["Rating predictor"] = {"MAE": mae, "RMSE": rmse}

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_rating_distribution(ratings, results_dir / "rating_distribution.png")
    plot_metric_comparison({
        name: value for name, value in metrics.items()
        if "precision_at_5" in value
    }, results_dir / "method_comparison.png")
    return train, test, metrics

