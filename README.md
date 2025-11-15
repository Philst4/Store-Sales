# Store Sales Forecasting Pipeline
End-to-End Machine Learning Project

This repository contains a fully reproducible ML pipeline for the **Kaggle Store Sales Forecasting** competition developed to be run via terminal. It demonstrates practical skills across **data engineering**, **ML model development**, **hyperparameter optimization**, **experiment reproducibility**, and **deployment-style automation**.

The project can be run via:

1. **Google Colab via Jupyter notbook** (easiest)
2. **Locally via terminal scripts**
3. **Locally via Jupyter notebook**

Notebooks mirror terminal/CLI.

---

## Project Overview

This pipeline covers the full workflow:

**Raw data → Data Preprocessing + Feature engineering → Hyperparameter tuning → Model training → Submission file generation**

Key components:

- **Data preprocessing & feature engineering**  
- **XGBoost / LightGBM model training**  
- **Optuna hyperparameter tuning**  
- **Modular, script-based architecture**  
- **Environment-managed reproducibility (conda)**  
- **Notebook and CLI execution**

This project reflects real-world workflows for **Data Science**, **Machine Learning Engineering**, and **Software Engineering** roles.

---

## Project Structure

```
Store-Sales/
├── README.md
├── environment.yml                  # Conda environment for reproducibility
├── config.yaml                      # Configuration file for paths/parameters
├── pipeline_local.ipynb             # Local notebook pipeline
├── pipeline_colab.ipynb             # Colab notebook pipeline
├── Screen Shot *.png                 # Example screenshots / visuals
│
├── data/                            # Dataset folders
│   ├── raw/                         # Raw dataset files
│   │   ├── holidays_events.csv
│   │   ├── oil.csv
│   │   ├── stores.csv
│   │   ├── test.csv
│   │   └── train.csv
│   └── clean/                       # Cleaned / processed datasets (generated)
│
├── scripts/                         # Script-based pipeline components
│   ├── load_from_manifest.py        # Utility for loading data
│   ├── make_submission.py           # Generate submission CSV
│   ├── open_notebook_in_colab.py    # Utility to open notebooks in Colab
│   ├── process_data.py              # Data cleaning & feature engineering
│   ├── train_best.py                # Train final model
│   └── tune_model.py                # Hyperparameter tuning with Optuna
│
├── src/                             # Core Python modules
│   ├── __init__.py
│   ├── data_processing.py           # Functions for processing / aggregating data
│   ├── io_utils.py                  # I/O utilities
│   ├── model_tuning.py              # Functions for hyperparameter tuning
│   └── modeling.py                  # Model training & evaluation
│
├── models/                           # Saved models (generated)
├── submissions/                      # Submission CSVs (generated)
├── experiment_configs/               # Configurations for model tuning + training
└── tests/                            # Unit tests / validation scripts
```

---

## Setup

### 1. Google Colab Setup with Notebook (Easiest)

1. Open **`pipeline_colab.ipynb`** in Colab.  
2. Run all cells — the notebook handles dependencies, dataset download, and pipeline execution.

_No local setup required._

---

### 2.  Local Setup with Terminal and/or Notebook

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

Can either run from terminal:

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

Or local notebook execution:
1. Open **`pipeline_local.ipynb`**   
2. Run cells top-to-bottom (mirrors terminal/CLI workflow)

Output directories:

```
data/clean/ # Cleaned + processed data
optuna_studies.db # Hyperparameter tuning trials
models/ # Trained model instances
submissions/ # Predictions model makes on test set
```

---

## Technologies Used

- Python  
- XGBoost / LightGBM  
- Optuna  
- Pandas / Dask
- NumPy  
- Matplotlib / Seaborn  
- Conda (reproducibility)  
- Jupyter / Google Colab

---

## Skills Demonstrated

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
- Script-driven workflow 

---

## Notes

- Small data sample included in GH repo so pipeline can run; full dataset excluded
- Full dataset publicly available and comes from **Kaggle Store Sales Forecasting** competition
- The scripts/notebooks download or generate necessary files automatically.

