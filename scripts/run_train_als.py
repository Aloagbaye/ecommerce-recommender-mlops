import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix
import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set tracking URI to local MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

INPUT_FILE = "data/processed/interactions.csv"
MODEL_OUTPUT = "models/als_model.pkl"
TOP_N = 10

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.models.als_pyfunc_model import ALSRecommender


def load_data():
    df = pd.read_csv(INPUT_FILE)
    return df

def build_matrix(df):
    user_item_matrix = coo_matrix(
        (df['rating'], (df['user_idx'], df['item_idx']))
    ).tocsr()
    return user_item_matrix

def get_top_n_recs(model, user_item_matrix, n=10):
    user_id = 0  # test on first user
    scores = model.recommend(user_id, user_item_matrix[user_id], N=n)
    return [int(item_id) for item_id in scores[0]]

def main():
    df = load_data()
    user_item_matrix = build_matrix(df)

    with mlflow.start_run(run_name="als_matrix_factorization"):

        mlflow.log_param("model", "ALS")
        mlflow.log_param("factors", 64)
        mlflow.log_param("iterations", 15)

        als_model = AlternatingLeastSquares(factors=64, iterations=15)
        als_model.fit(user_item_matrix.T.tocsr())  # implicit expects item-user matrix

        # Log sample top-N for user 0
        top_items = get_top_n_recs(als_model, user_item_matrix, TOP_N)
        mlflow.log_param("sample_user", 0)
        mlflow.log_param("sample_top_items", str(top_items))

        # Save model as a pickle
        with open(MODEL_OUTPUT, "wb") as f:
            pickle.dump(als_model, f)

        mlflow.log_artifact(MODEL_OUTPUT)

        # Optional: log dummy metric
        mlflow.log_metric("num_users", user_item_matrix.shape[0])
        mlflow.log_metric("num_items", user_item_matrix.shape[1])

        # Register the model (first time only)       
        mlflow.pyfunc.log_model(
    		artifact_path="als_model",
    		python_model=ALSRecommender(),
    		artifacts={"model_pickle": MODEL_OUTPUT}
		)

        print("Top-N recommended item IDs for user 0:", top_items)

if __name__ == "__main__":
    main()
