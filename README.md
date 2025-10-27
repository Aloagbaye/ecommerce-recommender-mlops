
# Ecommerce‑Recommender‑MLOps  
A production‑style end‑to‑end recommender‑system project built with data versioning, experiment tracking, drift monitoring, and deployment.  
GitHub: https://github.com/Aloagbaye/ecommerce-recommender-mlops  

---

## 🎯 Project Overview  
This project aims to build a **product recommendation system** for an e‑commerce platform that is:  
- **Reproducible** using data & model versioning (DVC)  
- **Trackable** using experiment logging and model registry (MLflow)  
- **Monitored** for data drift and concept drift over time (Evidently)  
- **Deployable** as a live service (API and/or UI)  

---

## 📁 Project Structure  
```
ecommerce-recommender-mlops/
├── data/
│   ├── raw/                 – raw input datasets  
│   └── processed/           – cleaned & transformed data  
├── notebooks/               – exploratory analysis & prototypes  
├── src/                     – modular source code (data, models, drift)  
├── scripts/                 – CLI scripts for pipeline stages  
├── models/                  – saved model artifacts  
├── outputs/                 – evaluation & drift report outputs  
├── mlruns/                  – MLflow tracking directory  
├── dvc.yaml                 – DVC pipeline definition  
├── requirements.txt         – Python dependencies  
└── .github/workflows/       – CI/CD workflows (GitHub Actions)  
```

---

## 🧭 Modules & Learning Path  
- **Module 1 – Project Setup & Problem Framing**  
  Set up environment, define business problem, initialize Git, DVC, MLflow.  
- **Module 2 – Baseline Recommender & MLflow Tracking**  
  Build a simple popularity‑based recommender and log experiments.  
- **Module 3 – Data Versioning with DVC**  
  Version datasets and define reproducible pipelines (`dvc.yaml`).  
- **Module 4 – Improved Model & MLflow Model Registry**  
  Train a matrix factorization model (with the `implicit` library), register it in MLflow.  
- **Module 5 – Drift Detection with Evidently**  
  Detect data and concept drift in user‑item interactions.  
- **Module 6 – Monitoring & CI/CD**  
  Automate drift detection and retraining with GitHub Actions.  
- **Module 7 – Serving Recommendations (API/UI)**  
  Deploy the model via API (FastAPI) or UI (Streamlit) and handle user input + feedback loop.

---

## 🔧 Getting Started  

### 1. Clone the repo  
```bash
git clone https://github.com/Aloagbaye/ecommerce-recommender-mlops.git  
cd ecommerce-recommender-mlops  
```  

### 2. Set up environment  
```bash
# Create virtual environment
python -m venv .venv  
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  
```  

### 3. Download & prepare data  
- Download the Amazon Reviews dataset (e.g., Electronics) to `data/raw/`.  
- Run preprocessing:  
```bash
python scripts/run_prepare_data.py  
```  
- Track data with DVC:  
```bash
dvc init  
dvc add data/processed/interactions.csv  
git commit -m "Track processed interactions with DVC"  
```  

### 4. Run baseline experiment  
```bash
python scripts/run_train_baseline.py  
```  
View results in MLflow UI at `http://localhost:5000`.

### 5. Train advanced model and register  
```bash
python scripts/run_train_als.py  
```  
Register the model via MLflow UI → Model Registry.

### 6. Detect drift  
```bash
python scripts/simulate_drift.py  
python scripts/run_drift_report.py  
```  
Artifacts and metrics logged to MLflow.

### 7. Serve recommendations (optional)  
```bash
uvicorn app.main:app --reload  
```  
Call endpoint (e.g., `POST /recommend`) with a user ID to get top‑N product recommendations.

---

## 🎥 Demo & Screenshots  
*(You may include links or screenshot images of the UI, drift report, MLflow dashboard here.)*

---

## 🛠 Key Technologies  
- **MLflow** – experiment tracking, model registry, model serving  
- **DVC** – data & model versioning, reproducible pipelines  
- **Implicit** – fast matrix factorization for implicit feedback  
- **Evidently** – data drift & concept drift detection  
- **FastAPI / Streamlit** – deployment & UI options  
- **GitHub Actions** – CI/CD automation  

---

## 🧠 Best Practices  
- Version all data artifacts and code scripts for reproducibility  
- Log hyperparameters, metrics, artifacts to MLflow for visibility  
- Monitor drift regularly — stale models degrade quickly  
- Automate pipeline runs and deployments to avoid manual errors  
- Containerize the service (Docker) for portability  

---

## ✅ Contributing  
Contributions are welcome! Please:  
- Fork the repo  
- Create a feature branch (`git checkout -b feature/my‑feature`)  
- Submit a pull request with tests / documentation updates  

---

## 📝 License  
This project is licensed under the MIT License — see the `LICENSE` file for details.

---

## 📬 Contact  
Maintainer: [Aloagbaye](https://github.com/Aloagbaye)  
Questions, suggestions or issues? Open an issue on this repo.

---
