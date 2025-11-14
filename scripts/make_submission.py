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
    load_config,
    load_experiment_config,
    load_and_merge_from_manifest
)
from src.modeling import (
    get_train,
    get_test,
    get_features,
)

def main(args):
    
    # Deal with config
    config = load_config(args.config_path)
    CLEAN_DATA_PATH = config['clean_data_path']
    OPTUNA_STUDIES_URI = config['optuna_studies_uri']
    MANIFEST_PATH = os.path.join(CLEAN_DATA_PATH, "manifest.json")
    MODELS_PATH = config['models_path']
    SUBMISSIONS_PATH = config['submissions_path']
    if not os.path.exists(SUBMISSIONS_PATH):
        os.mkdir(SUBMISSIONS_PATH)
    
    n_seeds = args.n_seeds
    
    # Load experiment_config
    experiment_config = load_experiment_config(
        args.experiment_config_path
    )
    study_name = experiment_config['study_name']
    
    # Load in study
    study = optuna.load_study(
        study_name=study_name,
        storage=OPTUNA_STUDIES_URI
    )

    # Extract best trial and corresponding params
    best_trial = study.best_trial
    best_params = best_trial.params
    best_params = experiment_config['add_constant_hyperparams'](
        best_params
    )
    
    # Load in/extract 'test' data using dask
    test_ddf = load_and_merge_from_manifest(
        MANIFEST_PATH,
        start_date="08-16-2017",
        end_date="08-31-2017"
    )
    test_df = test_ddf.compute()
    test_df = get_test(test_df)
    
    X_te = get_features(test_df)
    ids = test_df['id']
    
    for seed in range(n_seeds):
        model_name = f"{study_name}_model_{seed}.joblib"
        model_path = f"{MODELS_PATH}{model_name}"
        submission_path = f"{SUBMISSIONS_PATH}{study_name}_submission_{seed}.csv"
        print(f"Loading in {model_name}...")
        model = joblib.load(model_path)
        
        # Make predictions
        print("Making predictions...")
        log_predictions = model.predict(X_te) 
        predictions = np.expm1(log_predictions) # Un-log
        
        # Combine with id's, convert to pandas
        print("Making submission...")
        submission = pd.DataFrame({
            'id' : ids.astype(int),
            'sales' : predictions.astype(float)
        })
        
        # Save predictions
        print(f"Saving submission to '{submission_path}'...")
        submission.to_csv(submission_path, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to make submission")
    
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config.yaml",
        help="Path of config file to use"
    )
    
    parser.add_argument("--experiment_config_path", type=str, default="experiment_configs.xgb", help="Python module path to experiment config (e.g. experiment_configs.xgb.py)")    
    parser.add_argument("--n_seeds", type=int, default=1, help="Model suffix range to make submissions for")
    
    args = parser.parse_args()
    main(args)