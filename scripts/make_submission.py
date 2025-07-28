# STL imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse

# External imports
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client
from xgboost.dask import DaskXGBRegressor
import optuna

from src.io_utils import load_clean_data
from src.data_processing import merge_all
from src.modeling import (
    get_train,
    get_test,
    get_features,
)

def main(args):
    # From args/config
    study_name = "xgb"
    storage_uri = "sqlite:///./optuna_studies.db"
    clean_data_path = "./data/clean/"
    model_weights_path = "./dask_xgboost_model.json"
    submission_path = "./submission.csv"
    
    # Start dask client
    client = Client()
    
    # Load in data using dask
    dfs = load_clean_data(clean_data_path, as_dask=True)
    
    # Merge data
    test_data = merge_all(
        get_test(dfs['main']), # Extract test data
        dfs['stores'], 
        dfs['oil'], 
        dfs['holidays_events']
    )
    del dfs
    test_features = get_features(test_data)
    ids = test_data['id']
    del test_data
    
    
    print(f"Loading in model...")
    # Load in study
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_uri
    )

    # Extract best trial and corresponding params
    best_trial = study.best_trial
    best_params = best_trial.params
    
    # Make model
    model = DaskXGBRegressor(**best_params) # Needed to ensure model is set up the same.
    model.load_model(model_weights_path)
    
    # Make predictions
    print("Making predictions...")
    log_predictions = model.predict(test_features) # dask array
    log_predictions = log_predictions.compute() # pd.dataframe
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
    args = None
    main(args)