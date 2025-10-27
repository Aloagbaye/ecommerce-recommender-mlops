import mlflow.pyfunc
import pickle

class ALSRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model_pickle"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        user_ids = model_input["user_id"].values
        top_n = model_input.get("top_n", 10)
        recs = []
        for user_id in user_ids:
            user_recs = self.model.recommend(user_id, self.model.user_items[user_id], N=top_n)
            recs.append([item for item, score in user_recs])
        return recs