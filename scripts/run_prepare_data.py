# scripts/run_prepare_data.py

import os
import gzip
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_DATA_PATH = "data/raw/Electronics_5.json.gz"
PROCESSED_DIR = "data/processed/"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "interactions.csv")

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 10

def parse_gz(path):
    with gzip.open(path, 'rb') as f:
        for line in f:
            yield json.loads(line)

def load_data():
    print(f"Loading data from: {RAW_DATA_PATH}")
    data = list(parse_gz(RAW_DATA_PATH))
    df = pd.DataFrame(data)
    print(f"Original records: {len(df)}")
    return df

def clean_data(df):
    df = df[['reviewerID', 'asin', 'overall']].dropna()
    df.columns = ['user_id', 'item_id', 'rating']
    df = df[df['rating'] > 0]  # remove weird edge cases
    return df

def filter_active_users_items(df):
    user_counts = df['user_id'].value_counts()
    item_counts = df['item_id'].value_counts()

    df = df[df['user_id'].isin(user_counts[user_counts >= MIN_USER_INTERACTIONS].index)]
    df = df[df['item_id'].isin(item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index)]

    print(f"Filtered down to {len(df)} interactions")
    return df

def encode_ids(df):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df['user_idx'] = user_encoder.fit_transform(df['user_id'])
    df['item_idx'] = item_encoder.fit_transform(df['item_id'])

    return df[['user_idx', 'item_idx', 'rating']]

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_data()
    df = clean_data(df)
    df = filter_active_users_items(df)
    df = encode_ids(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved processed interactions to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
