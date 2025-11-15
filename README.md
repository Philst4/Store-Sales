# 📦 Store Sales Forecasting Pipeline
End-to-End Machine Learning Project

This repository contains a fully reproducible ML pipeline for the **Kaggle Store Sales Forecasting** competition. It demonstrates practical skills across **data engineering**, **ML model development**, **hyperparameter optimization**, **experiment reproducibility**, and **deployment-style automation**.

The project can be run via:

<<<<<<< HEAD
# Setting project up locally
- Clone project (maybe download git?)
- Navigate to project root directory in terminal
- Install conda before if necessary
- Make environment called 'store-sales' with environment.yml; activate it
=======
1. **Google Colab** (easiest)
2. **Locally via terminal scripts**
3. **Locally via Jupyter notebook**
>>>>>>> 73b7680990bf49b144b49d08050a9f5ab74f6729

---

## 🚀 Project Overview

<<<<<<< HEAD
# Running project from terminal (locally)
(1) python scripts/process_data.py
=======
This pipeline covers the full workflow:

**Raw data → Feature engineering → Hyperparameter tuning → Model training → Submission file generation**
>>>>>>> 73b7680990bf49b144b49d08050a9f5ab74f6729

Key components:

- **Data preprocessing & feature engineering**  
- **XGBoost / LightGBM model training**  
- **Optuna hyperparameter tuning**  
- **Modular, script-based architecture**  
- **Environment-managed reproducibility (conda)**  
- **Notebook and CLI execution**

This project reflects real-world workflows for **Data Science**, **Machine Learning Engineering**, and **Software Engineering** roles.

---

## 🧱 Project Structure

```
Store-Sales/
├── data/                 # Raw & processed datasets (ignored via .gitignore)
├── scripts/
│   ├── process_data.py   # Cleans + engineers features
│   ├── tune_model.py     # Runs Optuna study
│   ├── train_best.py     # Trains final model using best trial
│   └── make_submission.py
├── notebooks/
│   ├── pipeline_local.ipynb
│   └── pipeline_colab.ipynb
├── environment.yml
└── README.md
```

---

## 🛠️ Setup

### 1. 📥 Google Colab (Easiest)

1. Open **`pipeline_colab.ipynb`** in Colab.  
2. Run all cells — the notebook handles dependencies, dataset download, and pipeline execution.

_No local setup required._

---

### 2. 🖥️ Local Setup (Terminal)

#### Step 1 — Clone the repository

```bash
git clone https://github.com/Philst4/Store-Sales.git
cd Store-Sales
```

#### Step 2 — Create and activate conda environment

```bash
conda env create -f environment.yml
conda activate store-sales
```

#### Step 3 — Run the pipeline

```bash
# 1️⃣ Data processing & feature engineering
python scripts/process_data.py

# 2️⃣ Hyperparameter tuning with Optuna
python scripts/tune_model.py

# 3️⃣ Train best model
python scripts/train_best.py

# 4️⃣ Generate submission
python scripts/make_submission.py
```

Output directories:

```
data/processed/
models/
submissions/submission.csv
optuna_studies.db
```

---

### 3. 🧪 Notebook Execution (Local)

1. Open **`pipeline_local.ipynb`**  
2. Activate conda environment  
3. Run cells top-to-bottom (mirrors CLI workflow)

---

## 📊 Technologies Used

- Python  
- XGBoost / LightGBM  
- Optuna  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- Conda (reproducibility)  
- Jupyter / Google Colab

---

## 🎯 Skills Demonstrated

**Machine Learning:**

- Feature engineering  
- Hyperparameter optimization (Optuna)  
- Cross-validation & evaluation strategies  

**Software Engineering:**

- Modular, maintainable Python scripts  
- Reproducible environments  
- Clear project structure  
- Automated pipelines  

**End-to-End Deployment Thinking:**

- Reproducible local + cloud execution  
- Script-driven workflow (production-like)  
- Clear, professional pipeline for reviewers/interviews  

---

## 📝 Notes

- Dataset comes from **Kaggle Store Sales Forecasting** competition.  
- Data folders are excluded via `.gitignore`; the scripts/notebooks download or generate necessary files automatically.

<<<<<<< HEAD
(4) python scripts/make_submission.py

# Running project using local notebook 'pipeline_local.ipynb'
Run cells in notebook

# Running project using Colab notebook 'pipeline_colab.ipynb'
Run cells in notebook
=======
>>>>>>> 73b7680990bf49b144b49d08050a9f5ab74f6729
