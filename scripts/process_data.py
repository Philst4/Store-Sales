# STL imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse

# External imports
import numpy as np

# Internal imports
from src.io_utils import (
    load_config, 
    get_data_paths,
    load_raw_data,
    save_clean_data
)
from src.data_processing import (
    combine_train_test,
    process_data,
    assign_ascending_dates
)

def check_args(args):
    assert args.run_type in ("test", "production"), f"Unkown run_type '{args.run_type}'"
    assert args.compute_mode in ("local", "cloud"), f"Unknown compute_mode '{args.compute_mode}'"
    assert args.storage_mode in ("local", "cloud"), f"Unknown storage_mode '{args.storage_mode}'"

# Main function
def main(args):
    """
    Runs data processing pipeline.
    
    Loads in raw data to clean, feature engineer, and merge.
    Saves data thereafter.    
    """
    
    print("Running pipeline...")
    
    # Check args
    check_args(args)
    
    # Load config    
    config = load_config()
    
    # Get data paths
    RAW_DATA_PATH, CLEAN_DATA_PATH = get_data_paths(args.storage_mode, config)
    
    # Load in data
    dfs = load_raw_data(RAW_DATA_PATH)
    
    # Do processing
    dfs['main'], train_ids, test_ids = combine_train_test(
        dfs['train'],
        dfs['test']
    )
    del dfs['train']
    del dfs['test']
    
    dfs['main'], dfs['stores'], dfs['oil'], dfs['holidays_events'] = process_data(
        dfs['main'],
        dfs['stores'], 
        dfs['oil'], 
        dfs['holidays_events']
    )

    # Add 'is_train', 'is_test'
    dfs['main']['is_train'] = dfs['main']['id'].isin(train_ids)
    dfs['main']['is_test'] = dfs['main']['id'].isin(test_ids)
    
    # Only for 'tests'
    if args.run_type == "test":
        # Diversify dates
        #merged = assign_ascending_dates(merged)
        pass
    
    # Save data (along with category metadata)
    save_clean_data(CLEAN_DATA_PATH, dfs)
    pass
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing Script")
   
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
    
    args = parser.parse_args()
    main(args)
    print(f"Script complete")
