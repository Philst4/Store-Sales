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

# Internal imports
from src.io_utils import (
    load_config, 
    get_data_paths,
    load_raw_data,
    save_as_parquet,
    save_clean_data
)
from src.data_processing import (
    combine_train_test,
    process_main,
    process_stores,
    process_oil,
    process_holidays_events,
    compute_rolling_stats,
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
    
    # Initialize Dask client
    client = Client()
    
    # WILL BE ARGS
    windows_from_0 = [1, 2, 4, 7, 14]
    lag = 16
    windows_from_lag = [1, 7, 28, 91, 365]
    quantiles= [25, 50, 75]
    
    # Check args
    check_args(args)
    
    # Load config    
    config = load_config()
    
    # Get data paths
    RAW_DATA_PATH, CLEAN_DATA_PATH = get_data_paths(args.storage_mode, config)
    
    # Load in data
    # TODO CHANGE LOGIC HERE... LOAD IN EACH DATASET SEPARATELY
    
    # (1) Load/process/save train/test -> main
    print(f"Processing 'train'/'test' -> 'main'...")
    train = dd.read_csv(os.path.join(RAW_DATA_PATH, "train.csv"))
    test = dd.read_csv(os.path.join(RAW_DATA_PATH, "test.csv"))
    main, _, _ = combine_train_test(train, test)
    del train
    del test
    main = process_main(main)
    save_as_parquet(main, "main", CLEAN_DATA_PATH)
    del main
    
    # (2) Load/process/save stores
    print(f"Processing 'stores'...")
    stores = dd.read_csv(os.path.join(RAW_DATA_PATH, "stores.csv"))
    stores = process_stores(stores)
    save_as_parquet(stores, "stores", CLEAN_DATA_PATH)
    del stores
    
    # (3) Load/process/save oil
    print(f"Processing 'oil'...")
    oil = dd.read_csv(os.path.join(RAW_DATA_PATH, "oil.csv"))
    oil = process_oil(
        oil,
        windows_from_0=windows_from_0,
        lag=lag,
        windows_from_lag=windows_from_lag
    )
    save_as_parquet(oil, "oil", CLEAN_DATA_PATH)
    del oil
    
    # (4) Load/process/save holidays_events
    print(f"Processing 'holidays_events'...")
    holidays_events = dd.read_csv(os.path.join(RAW_DATA_PATH, "holidays_events.csv"))
    holidays_events = process_holidays_events(holidays_events)
    save_as_parquet(holidays_events, "holidays_events", CLEAN_DATA_PATH)
    del holidays_events
    
    # (5) Compute rolling stats using main + stores
    print(f"Computing rolling stats using 'main' and 'stores'...")
    
    # Load in both, merge
    main = dd.read_parquet(os.path.join(CLEAN_DATA_PATH, "main.parquet"), engine="pyarrow")
    stores = dd.read_parquet(os.path.join(CLEAN_DATA_PATH, "stores.parquet"), engine="pyarrow")
    main_stores = dd.merge(
        main,
        stores,
        on='store_nbr',
        how='left'
    ).set_index('date')
    del main
    del stores
    
    # Get groups
    main_stores_groups = [
        [cat_col] for cat_col in 
        list(main_stores.select_dtypes(exclude='number').columns)
        if cat_col not in ('date', 'is_train', 'is_test')
    ]
    main_stores_groups.append([])
    
    # Calculate rolling stats wrt each group, window
    for group_cols in main_stores_groups:
        suffix = f"_wrt_{'_'.join(group_cols)}" if group_cols else ""
        for window in windows_from_lag:
            
            # Calculate rolling stats
            rolling_stats = compute_rolling_stats(
                main_stores, 
                group_cols=group_cols,
                rolling_cols=['sales', 'log_sales'],
                lag=lag,
                window=window,
                quantiles=quantiles,
                suffix=suffix
            )
            
            # Save rolling stats
            file_name = f"rolling_lag{lag}_window{window}{suffix}"
            save_as_parquet(rolling_stats, file_name, CLEAN_DATA_PATH)
            #del rolling_stats

    return
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
        dfs['holidays_events'],
        windows_from_0=args.windows_from_0,
        lag=args.lag,
        windows_from_lag=args.windows_from_lag,
        quantiles=args.quantiles
    )

    # Add 'is_train', 'is_test'
    dfs['main'] = dfs['main'].assign(
        is_train=dfs['main']['id'].isin(train_ids),
        is_test=dfs['main']['id'].isin(test_ids)
    )
    
    # Only for 'tests'
    if args.run_type == "test":
        # Diversify dates
        # merged = assign_ascending_dates(merged)
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
    
    parser.add_argument(
        "--windows_from_0",
        type=float,
        nargs='*', # 0 or more
        default=[1, 2, 4, 7, 14],
        help="Which windows to take rolling stats from starting from 0 (used only for oil prices)"
    )
    
    parser.add_argument(
        "--lag",
        type=float,
        default=16,
        help= (
            "How much lag rolling stats should have (e.g. if we want" 
            "to make rolling stats useful for 16 days out, we should" 
            "have 'lag=16')."
        ),
    )
    
    parser.add_argument(
        "--windows_from_lag",
        type=float,
        nargs='*', # 0 or more 
        default=[1, 7, 14, 28, 91, 365],
        help="Which windows to take rolling stats from, starting from 'lag'"
    )
    
    
    parser.add_argument(
        "--quantiles",
        type=float, 
        nargs='*', # 0 or more
        default=[50],
        help="Which quantiles to include in rolling stats (between 0 and 100)"
    )
    
    args = parser.parse_args()
    main(args)
    print(f"Script complete")
