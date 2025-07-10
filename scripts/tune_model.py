import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse

# Internal imports
from src.io_utils import (
    load_config, 
    get_data_paths,
    load_clean_data,
    load_experiment_config,
    get_studies_uri
)

from src.data_processing import merge_all

from src.model_tuning import (
    get_train,
    make_objective,
    run_experiment
)

def main(args):
    """
    Main function for script for tuning a specified model architecture.
    
    Given an experiment config, this script tunes a model according to
    its specifications. 
    """
    
    # Load in config
    config = load_config()
    
    # Get data path, study_uri
    _, CLEAN_DATA_PATH = get_data_paths(args.storage_mode, config)
    STUDIES_URI = get_studies_uri(args.storage_mode, config)
    
    # Load in clean data (dictionary of dfs)
    clean_dfs = load_clean_data(CLEAN_DATA_PATH)
    
    # Load in experiment configuration
    experiment_config = load_experiment_config(
        args.experiment_config
    )
    
    # So... eventually we want to do some merging on the fly for performance?
    # For now, we just merge everything and go
    merged_df = merge_all(
        clean_dfs['main'],
        clean_dfs['stores'],
        clean_dfs['oil'],
        clean_dfs['holidays_events']
    )
    
    # Divvy up data
    train_df = get_train(merged_df).sort_values(by=['date'])
    
    # Figure out n_jobs
    n_jobs = min(args.n_jobs, os.cpu_count())
    
    # Make the objective function
    objective = make_objective(
        train_df,
        n_jobs=n_jobs,
        **experiment_config
    )
    
    # Run experiment
    run_experiment(
        objective,
        args.n_trials,
        STUDIES_URI,
        experiment_config['experiment_name']
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Tuning Script")
    
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
    parser.add_argument("--n_trials", type=int, default=5, help="Number of study trials to run (e.g. 5)")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel (e.g. 2)")
    args = parser.parse_args()
    main(args)
    print(f"Script complete")