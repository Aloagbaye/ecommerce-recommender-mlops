import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import Workspace
import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5000")

BASE_FILE = "data/processed/interactions.csv"
DRIFT_FILE = "data/processed/interactions_drifted.csv"
OUTPUT_HTML = "outputs/drift_report.html"


def generate_report():
    base = pd.read_csv(BASE_FILE)
    drifted = pd.read_csv(DRIFT_FILE)

    report = Report(metrics=[DataDriftPreset()])
    updated_report = report.run(reference_data=base, current_data=drifted)

    os.makedirs("outputs", exist_ok=True)
    # ws.add_run(project.id, regular_snapshot)
    updated_report.save_html(OUTPUT_HTML)

    with mlflow.start_run(run_name="drift_detection"):
        mlflow.log_artifact(OUTPUT_HTML)

    print(f"Drift report saved and logged to MLflow at {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_report()
