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


from src.io_utils import load_clean_data
from src.data_processing import merge_all
from src.modeling import (
    get_train,
    get_test,
    get_features,
    get_targets
)

def main(args):
    # From args/config
    study_name = "xgb"
    storage_uri = "sqlite:///./optuna_studies.db"
    clean_data_path = "./data/clean/"
    model_weights_path = "./dask_xgboost_model.json"
    
    # Start dask client
    client = Client()
    
    # Load in data using dask
    dfs = load_clean_data(clean_data_path, as_dask=True)
    
    # Merge data
    training_data = merge_all(
        get_train(dfs['main']), # Extract training data
        dfs['stores'], 
        dfs['oil'], 
        dfs['holidays_events']
    )
    del dfs
    
    # Load in study
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_uri
    )

    # Extract best trial and corresponding params
    best_trial = study.best_trial
    best_params = best_trial.params
    
    # Enable categorical support (needed for Dask XGBRegressor)
    best_params['enable_categorical'] = True
    
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

    # Make model
    X_tr = get_features(training_data)
    y_tr = get_targets(training_data)
    model = DaskXGBRegressor(**best_params)
    print("Fitting model...")
    model.fit(X_tr, y_tr)
    print("Done fitting!")
    model.save_model(model_weights_path)
    print(f"Model saved to '{model_weights_path}'")
    
if __name__ == "__main__":
    args = None
    main(args)
    