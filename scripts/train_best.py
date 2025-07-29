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

def configure_client():
    """Configure Dask client with memory-aware settings"""
    return Client(
        n_workers=4,                   # Reduced from default
        threads_per_worker=1,          # Fewer threads per worker
        memory_limit='6GB',            # Hard limit per worker
        local_directory='/tmp/dask',   # Spill directory
        memory_target_fraction=0.6,    # Start spilling earlier
        memory_spill_fraction=0.7,     # Aggressive spilling
        memory_pause_fraction=0.8      # Pause at 80% memory
    )

def optimize_model_params(params):
    """Add memory-efficient XGBoost parameters"""
    params.update({
        'tree_method': 'hist',          # Required for categorical
        'single_precision_histogram': True,  # Less memory usage
        'max_bin': 256,                 # Fewer bins = less memory
        'predictor': 'cpu_predictor',   # Avoid GPU memory issues
        'enable_categorical': True      # Already in your code
    })
    return params

def main(args):
    # From args/config
    study_name = "xgb"
    storage_uri = "sqlite:///./optuna_studies.db"
    clean_data_path = "./data/clean/"
    model_weights_path = "./dask_xgboost_model.json"
    
    # Start dask client
    client = configure_client()
    print(client)
    
    try: 
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
        best_params = optimize_model_params(best_params)

        # Prepare data
        X_tr = get_features(training_data)
        y_tr = get_targets(training_data)
        
        # Persist only what's needed
        X_tr, y_tr = client.persist([X_tr, y_tr])
        
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
    