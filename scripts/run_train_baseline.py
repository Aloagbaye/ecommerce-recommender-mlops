# scripts/run_train_baseline.py

import pandas as pd
import mlflow
import os
from collections import Counter

INPUT_FILE = "data/processed/interactions.csv"
TOP_N = 10
ARTIFACTS_DIR = "outputs/"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
mlflow.set_tracking_uri("http://localhost:5000")

def load_data():
    df = pd.read_csv(INPUT_FILE)
    return df

def popularity_recommender(df, top_n=10):
    popular_items = df['item_idx'].value_counts().head(top_n).index.tolist()
    return popular_items

def evaluate(popular_items, df):
    unique_users = df['user_idx'].unique()[:100]
    hit_count = 0
    for user in unique_users:
        user_items = df[df['user_idx'] == user]['item_idx'].tolist()
        hit_count += int(any(item in user_items for item in popular_items))
    hit_rate = hit_count / len(unique_users)
    return hit_rate

def save_artifacts(recs):
    output_path = os.path.join(ARTIFACTS_DIR, "top_items.txt")
    with open(output_path, "w") as f:
        for item in recs:
            f.write(f"{item}\n")
    return output_path

def main():
    df = load_data()

    with mlflow.start_run(run_name="baseline_popularity"):
        mlflow.log_param("model", "popularity_based")
        mlflow.log_param("top_n", TOP_N)

        top_items = popularity_recommender(df, top_n=TOP_N)
        print("popular items saved")
        hit_rate = evaluate(top_items, df)
        print("hit rate")

        mlflow.log_metric("hit_rate", hit_rate)

        artifacts_path = save_artifacts(top_items)
        mlflow.log_artifact(artifacts_path)

        print(f"Top-{TOP_N} Hit Rate: {hit_rate:.4f}")

if __name__ == "__main__":
    main()
