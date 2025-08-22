import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse

# Internal imports
from src.io_utils import (
    load_config, 
    get_data_paths,
    load_and_merge_from_manifest,
    load_experiment_config,
    get_studies_uri
)

from src.modeling import get_train

from src.model_tuning import (
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
    
    
    # Load in data, split off training data (as pandas)
    print(f"Loading training data...")
    ddf = load_and_merge_from_manifest(
        "./data/clean/manifest.json", 
        sample=args.sample
    )
    
    print(f"Loading training data into memory...")
    train_df = get_train(ddf).compute()
    
    # Load in experiment configuration
    experiment_config = load_experiment_config(
        args.experiment_config
    )
    del experiment_config['add_constant_hyperparams']
    
    # Figure out n_jobs
    n_jobs = min(args.n_jobs, os.cpu_count())
    
    # Make the objective function
    objective = make_objective(
        train_df,
        n_backtests=args.n_backtests,
        valset_size=args.valset_size,
        n_jobs=n_jobs,
        max_time_per_trial=args.max_time_per_trial,
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
    
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of training samples to take from training data.")
    parser.add_argument("--experiment_config", type=str, default="experiment_configs.xgb", help="Python module path to experiment config (e.g. experiment_configs.xgb.py)")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of study trials to run (e.g. 5)")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel (e.g. 2)")
    parser.add_argument("--n_backtests", type=int, default=8, help="Number of backtests to run for each trial")
    parser.add_argument("--valset_size", type=int, default=16, help="Number of days included in each valset")
    args = parser.parse_args()
    main(args)
    print(f"Script complete")