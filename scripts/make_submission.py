# STL imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse
import joblib

# External imports
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client
import optuna

from src.io_utils import (
    load_experiment_config,
    load_and_merge_from_manifest
)
from src.modeling import (
    get_train,
    get_test,
    get_features,
)

def main(args):
    
    # From args/config
    study_name = "xgb"
    experiment_config = args.experiment_config
    studies_uri = "sqlite:///./optuna_studies.db"
    manifest_path = "./data/clean/manifest.json"
    model_path = "./model.joblib"
    submission_path = "./submission.csv"
    
    # Load in study
    study = optuna.load_study(
        study_name=study_name,
        storage=studies_uri
    )

    # Extract best trial and corresponding params
    experiment_config = load_experiment_config(
        experiment_config
    )
    best_trial = study.best_trial
    best_params = best_trial.params
    best_params = experiment_config['add_constant_hyperparams'](
        best_params
    )
    
    # Load in data using dask
    test_ddf = load_and_merge_from_manifest(
        manifest_path,
        start_date="08-16-2017",
        end_date="08-31-2017"
    )
    test_df = test_ddf.compute()
    test_df = get_test(test_df)
    
    X_te = get_features(test_df)
    ids = test_df['id']
    
    
    print(f"Loading in model...")
    model = joblib.load(model_path)
    
    # Make predictions
    print("Making predictions...")
    log_predictions = model.predict(X_te) 
    predictions = np.expm1(log_predictions) # Un-log
    
    # Combine with id's, convert to pandas
    print("Making submission...")
    submission = pd.DataFrame({
        'id' : ids,
        'sales' : predictions
    })
    
    # Save predictions
    print(f"Saving submission to '{submission_path}'...")
    submission.to_csv(submission_path, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to make submission")
    
    parser.add_argument("--experiment_config", type=str, default="experiment_configs.xgb", help="Python module path to experiment config (e.g. experiment_configs.xgb.py)")    
    
    
    args = parser.parse_args()
    main(args)