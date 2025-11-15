🧱 Project Structure
project/
│
├── data/                # Raw & processed datasets (ignored via .gitignore)
├── scripts/
│   ├── process_data.py  # Cleans + engineers features
│   ├── tune_model.py    # Runs Optuna study
│   ├── train_best.py    # Trains final model using best trial
│   └── make_submission.py
│
├── notebooks/
│   ├── pipeline_local.ipynb
│   └── pipeline_colab.ipynb
│
├── environment.yml
└── README.md
🛠️ Setup
1. 📥 Running on Google Colab (Easiest)
Open pipeline_colab.ipynb in Colab
Run all cells — the notebook handles:
Installing dependencies
Downloading dataset
Running the full pipeline
No local setup required.
2. 🖥️ Running Locally (Terminal Workflow)
Step 1 — Clone the repository
git clone https://github.com/Philst4/Store-Sales.git
cd Store-Sales
Step 2 — Create and activate the conda environment
conda env create -f environment.yml
conda activate store-sales
Step 3 — Run the full pipeline
# (1) Data processing & feature engineering
python scripts/process_data.py

# (2) Hyperparameter tuning with Optuna
python scripts/tune_model.py

# (3) Train best model found by tuning
python scripts/train_best.py

# (4) Generate submission.csv
python scripts/make_submission.py
This produces:
data/processed/
models/
submissions/submission.csv
optuna_studies.db
3. 🧪 Running Locally via Notebook
If you prefer a notebook workflow:
Open pipeline_local.ipynb
Ensure your conda environment is active
Run the notebook top-to-bottom
(it mirrors the CLI pipeline)
📊 Technologies Used
Python
XGBoost / LightGBM
Optuna (hyperparameter optimization)
Pandas / NumPy
Matplotlib / Seaborn
Conda (reproducibility)
Jupyter / Google Colab
🎯 Skills Demonstrated
This project showcases abilities valued in Data Science, Machine Learning Engineering, and Software Engineering roles:
🧠 Machine Learning
Advanced feature engineering
Model tuning with Bayesian optimization
Cross-validation and evaluation strategies
🏗️ Software Engineering
Modular, maintainable Python scripts
Reproducible environments
Clear project structure
Automated pipelines
📦 End-to-End Deployment Thinking
Full reproducibility (local + cloud)
Clear execution pathways for technical interview reviewers
Script-driven workflow that reflects production pipelines
📝 Notes
The dataset comes from the Kaggle Store Sales Forecasting competition.
Data folders are excluded via .gitignore; users must download them automatically via script/notebook.
