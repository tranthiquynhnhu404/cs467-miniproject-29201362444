import argparse
import sys
from pathlib import Path

import pandas as pd

from nlp_pipeline import GiftNLPPipeline
from recommender import GiftRecommender
from evaluate import evaluate


ROOT = Path(__file__).resolve().parents[1]
ITEMS_PATH = ROOT / "data" / "items.csv"
RATINGS_PATH = ROOT / "data" / "ratings.csv"
RESULTS_DIR = ROOT / "results"


def format_recommendations(title, recommendations):
    lines = [title]
    if not recommendations:
        lines.append("  No recommendations found.")
        return "\n".join(lines)
    for rank, rec in enumerate(recommendations, 1):
        keywords = ", ".join(rec["keywords"]) if rec["keywords"] else "no strong keyword match"
        lines.append(
            f"{rank}. {rec['gift_name']} [{rec['item_id']}] "
            f"- score={rec['score']:.4f} - keywords: {keywords}"
        )
    return "\n".join(lines)


def run_demo():
    items = pd.read_csv(ITEMS_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    nlp = GiftNLPPipeline()
    recommender = GiftRecommender(items, ratings, nlp)

    outputs = []
    outputs.append(format_recommendations(
        "Test 1 - Occasion query: thoughtful handmade birthday gift for tea lover",
        recommender.hybrid_recommend(
            query="thoughtful handmade birthday gift for tea lover", top_n=5
        )
    ))

    bestseller = (
        ratings.groupby("item_id")["rating"].mean()
        .sort_values(ascending=False).index[0]
    )
    bestseller_row = items.loc[items["item_id"] == bestseller].iloc[0]
    outputs.append(format_recommendations(
        f"Test 2 - Similar gift alternatives to bestseller {bestseller_row['gift_name']} [{bestseller}]",
        recommender.similar_items(bestseller, top_n=5)
    ))

    recommender.add_cold_start_item(
        item_id="G041",
        gift_name="Seasonal Handmade Tea Advent Calendar",
        description="handmade winter tea advent calendar with thoughtful holiday tea blends for a tea lover",
        occasion_tags="christmas holiday winter",
        recipient_tags="tea lover seasonal gift",
    )
    outputs.append(format_recommendations(
        "Test 3 - Cold-start discovery: brand-new seasonal gift",
        recommender.content_recommend_query(
            "handmade tea holiday gift for tea lover", top_n=5
        )
    ))

    train, test, metrics = evaluate(items, ratings, RESULTS_DIR)
    output_text = "\n\n".join(outputs)
    output_text += "\n\nEvaluation\n"
    for name, values in metrics.items():
        output_text += f"{name}: " + ", ".join(
            f"{key}={value:.4f}" for key, value in values.items()
        ) + "\n"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "sample_outputs.txt").write_text(output_text, encoding="utf-8")

    print(output_text)
    print(f"\nCharts saved in: {RESULTS_DIR}")


def interactive():
    items = pd.read_csv(ITEMS_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    recommender = GiftRecommender(items, ratings, GiftNLPPipeline())

    parser = argparse.ArgumentParser(description="Gift recommender demo")
    parser.add_argument("--user", help="User ID, for example U01")
    parser.add_argument("--query", help="Free-text gift query")
    parser.add_argument("--similar", help="Item ID for similar-gift search")
    args = parser.parse_args()

    if args.similar:
        print(format_recommendations(
            f"Similar gifts to {args.similar}",
            recommender.similar_items(args.similar, top_n=5),
        ))
    elif args.query:
        print(format_recommendations(
            f"Query results for: {args.query}",
            recommender.hybrid_recommend(query=args.query, top_n=5),
        ))
    elif args.user:
        print(format_recommendations(
            f"Recommendations for {args.user}",
            recommender.hybrid_recommend(user_id=args.user, top_n=5),
        ))
    else:
        run_demo()


if __name__ == "__main__":
    interactive()
