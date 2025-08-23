# STL imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse
import joblib

# External imports
import optuna
import pandas as pd
import numpy as np

import xgboost as xgb

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster


from src.io_utils import (
    load_and_merge_from_manifest,
    load_experiment_config
)

from src.modeling import (
    get_train,
    get_test,
    get_features,
    get_targets
)

from sklearn.metrics import root_mean_squared_error

def configure_client():
    cluster = LocalCluster(
        n_workers=6,
        threads_per_worker=1,
        memory_limit=40 / 6,
        silence_logs=False
    )
    return Client(cluster)

def main(args):
    rng = np.random.default_rng(seed=42)
    
    # From args/config
    study_name = "xgb"
    experiment_config = args.experiment_config
    studies_uri = "sqlite:///./optuna_studies.db"
    manifest_path = "./data/clean/manifest.json"
    model_path = "./model.joblib"
    
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
        
    # Print info
    print(f"\n--- Training using following trial.... ---")
    print("Best trial number:", best_trial.number)
    print("Best value (objective/loss):", best_trial.value)
    print("Best hyperparameters:")
    for key, value in best_params.items():
        print(f" * {key}: {value}")
    print()
    
    # Initialize model
    model = None
    
    # Start dask client
    client = Client()
    
    #### FOR SAMPLING WITH REPLACEMENT ####
    n_iter = args.n_iter
    sample_frac = args.sample_frac
    for i in range(n_iter):
        print(f"\n -- Training Iteration {i+1}/{n_iter} (sampling {sample_frac * 100:.2f}% of data) --")
        
        # Load in data using dask
        ddf = load_and_merge_from_manifest(
            manifest_path,
            sample=sample_frac
        )
        print(f"Loading chunk into memory...")
        df = ddf.compute()
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')
            
        if len(df) == 0:
            print("Empty chunk")
        else:
            # Prepare data
            print(f"Splitting train/test...")
            training_df = get_train(df)
            X_tr = get_features(training_df)
            y_tr = get_targets(training_df)
            
            print("Training model on chunk...")

            if model is None:
                # Set early stopping rounds to False
                best_params['early_stopping_rounds'] = False
                
                # Create the pipeline model with the first chunk's schema
                model = experiment_config['build_model'](
                    X_tr, 
                    best_params
                )
                model.fit(X_tr, y_tr)
            else:
                # Access the XGBRegressor inside the pipeline
                xgb_step = model.named_steps['model']
                    
                # Fit again with xgb_model to continue training
                xgb_step.fit(
                    model.named_steps['preprocessor'].transform(X_tr),
                    y_tr,
                    xgb_model=xgb_step.get_booster()
                )
                
            print(f"Loss on chunk: {root_mean_squared_error(model.predict(X_tr), y_tr)}")
            
    print("Done training!")
    joblib.dump(model, model_path)
    print(f"Model saved to '{model_path}'")
    client.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script Using Best Hyperparameters")
    
    parser.add_argument(
        "--run_type", 
        type=str, 
        default="test", 
        help="Type of run (test or production)"
    ) 
    
    parser.add_argument(
        "--compute_mode",
        type=str,
        default="local",
        help="Where/how script is running (local or cloud)"
    )
    
    parser.add_argument(
        "--storage_mode",
        type=str,
        default="local",
        help="Where things are stored relative to script (local or cloud)"
    )
    
    parser.add_argument("--experiment_config", type=str, default="experiment_configs.xgb", help="Python module path to experiment config (e.g. experiment_configs.xgb.py)")    
    parser.add_argument("--sample_frac", type=float, default=0.1, help="The fraction of samples to include in a batch.")
    parser.add_argument("--n_iter", type=int, default=10, help="The number of batches to load/train on.")
    
    args = parser.parse_args()
    
    main(args)
    