
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

## 🎥 MLFlow and Evidently Screenshots 
<img width="1915" height="597" alt="image" src="https://github.com/user-attachments/assets/2af9ed84-5720-472a-9742-5d2e257e5f7c" />
- **Data Drift with Evidently**
  <img width="1838" height="781" alt="image" src="https://github.com/user-attachments/assets/fd4bdb04-2fb2-43f2-afa1-0c21c0cf9889" />

  <img width="1837" height="823" alt="image" src="https://github.com/user-attachments/assets/c833749e-5596-4d90-9a53-dc951b330442" />


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
