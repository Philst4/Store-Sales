# STL imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse

# External imports

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
    merge_all,
    assign_ascending_dates
)

# Main function
def main(args):
    """
    Runs data processing pipeline.
    
    Loads in raw data to clean, feature engineer, and merge.
    Saves data thereafter.    
    """
    
    print("Running pipeline...")
    
    # Load config    
    config = load_config()
    
    # Get data paths
    RAW_DATA_PATH, CLEAN_DATA_PATH = get_data_paths(args.mode, config)
    
    # Load in data
    train, test, stores, oil, holidays_events = load_raw_data(RAW_DATA_PATH)
    
    # Do processing
    main, train_ids, test_ids = combine_train_test(train, test)
    main, stores, oil, holidays_events = process_data(
        main,
        stores, 
        oil, 
        holidays_events
    )
    merged = merge_all(main, stores, oil, holidays_events)

    # Add 'is_train', 'is_test'
    merged['is_train'] = merged['id'].isin(train_ids)
    merged['is_test'] = merged['id'].isin(test_ids)
    
    # Only for 'tests'
    if args.mode in ('test_local', 'test_cloud'):
        # Diversify dates
        merged = assign_ascending_dates(merged)
    
    # Save data (along with category metadata)
    save_clean_data(CLEAN_DATA_PATH, merged)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing Script")
    parser.add_argument("--mode", type=str, default="test_local", help="Mode to run script in (test_local, or test_cloud, or production)") 
    args = parser.parse_args()
    assert args.mode in ("test_local", "test_cloud", "test_production")
    main(args)
    print(f"Script complete")
