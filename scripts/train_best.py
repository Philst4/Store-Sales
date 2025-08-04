# STL imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse

# External imports
import optuna
import dask.dataframe as dd
from dask.distributed import Client
from xgboost.dask import DaskXGBRegressor
from dask.diagnostics import ProgressBar


from src.io_utils import load_and_merge_from_manifest
from src.modeling import (
    get_train,
    get_test,
    get_features,
    get_targets
)

def configure_client():
    return Client()

def main(args):
    # From args/config
    study_name = "xgb"
    storage_uri = "sqlite:///./optuna_studies.db"
    manifest_path = "./data/clean/manifest.json"
    model_weights_path = "./dask_xgboost_model.json"
    
    # Start dask client
    client = configure_client()
    
    try: 
        # Load in data using dask
        ddf = load_and_merge_from_manifest(manifest_path)
        training_ddf = get_train(ddf)
        
        # Load in study
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_uri
        )

        # Extract best trial and corresponding params
        best_trial = study.best_trial
        best_params = best_trial.params
        
        # Print info
        print(f"--- Training using following trial.... ---", "\n")
        print("Best trial number:", best_trial.number)
        print("Best value (objective/loss):", best_trial.value)
        print("Best hyperparameters:")
        for key, value in best_params.items():
            print(f" * {key}: {value}")
        print()

        # Extract params from best trial
        best_params = best_trial.params

        # Prepare data
        print(f"Splitting train/test...")
        X_tr = get_features(training_ddf)
        y_tr = get_targets(training_ddf)
        
        # Persist only what's needed
        print(f"We get here")
        with ProgressBar():
            X_tr, y_tr = client.persist([X_tr, y_tr])
            print(f"We don't get here")

        model = DaskXGBRegressor(**best_params)
        print("Fitting model...")
        model.fit(
            X_tr, 
            y_tr,
            eval_set=[(X_tr, y_tr)]  # Monitor training
        )
        print("Done fitting!")
        model.save_model(model_weights_path)
        print(f"Model saved to '{model_weights_path}'")
    
    finally:
        # Ensure clean shutdown
        client.close()
        print("Dask client closed")
    
if __name__ == "__main__":
    args = None
    main(args)
    