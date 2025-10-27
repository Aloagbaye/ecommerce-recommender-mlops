
# 🚀 MLflow Cheatsheet

MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment.

---

## 🔧 1. Setup

```bash
pip install mlflow
mlflow ui  # Start MLflow tracking server at http://localhost:5000
```

---

## 📍 2. Set Tracking URI

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
```

---

## 🧪 3. Start and End Runs

```python
with mlflow.start_run(run_name="my_experiment"):
    mlflow.log_param("param1", 10)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("outputs/result.txt")
```

---

## 🧾 4. Log Parameters, Metrics, Artifacts

```python
mlflow.log_param("learning_rate", 0.01)
mlflow.log_metric("rmse", 0.89)
mlflow.log_artifact("models/model.pkl")
```

---

## 📦 5. Log Models

```python
import mlflow.sklearn

model = SomeSklearnModel()
mlflow.sklearn.log_model(model, artifact_path="sklearn-model")
```

---

## 🔁 6. Load Logged Models

```python
model = mlflow.sklearn.load_model("runs:/<run_id>/sklearn-model")
```

---

## 🔍 7. Search Runs

```python
runs = mlflow.search_runs(experiment_ids=["0"], filter_string="metrics.rmse < 1")
```

---

## 🧠 8. Custom Python Models

```python
import mlflow.pyfunc

class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model_input * 2

mlflow.pyfunc.log_model("custom-model", python_model=MyModel())
```

---

## 📊 9. Visual Interface

- Start the UI with `mlflow ui`
- Visit [http://localhost:5000](http://localhost:5000)

---

## 🗂️ 10. Experiments

```python
mlflow.create_experiment("my-experiment")
mlflow.set_experiment("my-experiment")
```

---

## 🧼 11. Clean Up (CLI)

```bash
mlflow experiments delete <experiment_id>
mlflow runs delete <run_id>
```

---

## 📚 Resources

- Docs: https://mlflow.org/docs/latest/index.html
- GitHub: https://github.com/mlflow/mlflow

---

✅ Keep this handy while building and tracking ML experiments!
