# STL imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse
import json
from time import time

# External imports
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client

# Internal imports
from src.io_utils import (
    load_config, 
    get_data_paths,
    save_as_parquet,
    save_cat_meta,
)
from src.data_processing import (
    combine_train_test,
    process_main,
    process_stores,
    process_oil,
    process_holidays_events,
    compute_rolling_stats,
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
    
    # Initialize Dask client
    client = Client()
    
    # WILL BE ARGS
    windows_from_0 = [1, 2, 4, 7, 14]
    lag = 16
    windows_from_lag = [1, 7, 28, 91, 365]
    supported_stats = ['mean', 'std', 'min', 'max']
    quantiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    quantiles = ['q' + str(quantile) for quantile in quantiles]
    
    # Load config    
    config = load_config(args.config_path)
    
    # Get data paths
    RAW_DATA_PATH = config['raw_data_path'] # should exist and have data
    CLEAN_DATA_PATH = config['clean_data_path']
    
    if not os.path.exists(CLEAN_DATA_PATH):
        os.makedirs(CLEAN_DATA_PATH)
    
    # Load in data
    
    # Keep track of a manifest file
    manifest = {
        "main_data" : None,
        "secondary_data" : [],
        "rolling_stats" : [],
    }
    
    
    # (1) Load/process/save train/test -> main; add to manifest
    print(f"Processing 'train'/'test' -> 'main'...")
    train = dd.read_csv(os.path.join(RAW_DATA_PATH, "train.csv"))
    test = dd.read_csv(os.path.join(RAW_DATA_PATH, "test.csv"))
    main, _, _ = combine_train_test(train, test)
    main = process_main(main)
    parquet_path = os.path.join(CLEAN_DATA_PATH, "main.parquet")
    save_as_parquet(main, parquet_path)
    cat_meta_path = os.path.join(CLEAN_DATA_PATH, "main_cat_meta.json")
    save_cat_meta(main, cat_meta_path)
    manifest['main_data'] = (
        {
            "name" : "main",
            "parquet_path" : parquet_path,
            "cat_meta_path" : cat_meta_path,
            "merge_cols" : []
        }
    )
    
    # (2) Load/process/save stores; track in manifest
    print(f"Processing 'stores'...")
    stores = dd.read_csv(os.path.join(RAW_DATA_PATH, "stores.csv"))
    stores = process_stores(stores)
    parquet_path = os.path.join(CLEAN_DATA_PATH, "stores.parquet")
    save_as_parquet(stores, parquet_path)
    cat_meta_path = os.path.join(CLEAN_DATA_PATH, "stores_cat_meta.json")
    save_cat_meta(stores, cat_meta_path)
    manifest['secondary_data'].append(
        {
            "name" : "stores",
            "parquet_path" : parquet_path,
            "cat_meta_path" : cat_meta_path,
            "merge_cols" : ["store_nbr"]
        }
    )
    
    # (3) Load/process/save oil; track in manifest
    print(f"Processing 'oil'...")
    oil = dd.read_csv(os.path.join(RAW_DATA_PATH, "oil.csv"))
    oil = process_oil(
        oil,
        windows_from_0=windows_from_0,
        lag=lag,
        windows_from_lag=windows_from_lag
    )
    parquet_path = os.path.join(CLEAN_DATA_PATH, "oil.parquet")
    save_as_parquet(oil, parquet_path)
    cat_meta_path = os.path.join(CLEAN_DATA_PATH, "oil_cat_meta.json")
    save_cat_meta(oil, cat_meta_path)
    manifest['secondary_data'].append(
        {
            "name" : "oil",
            "parquet_path" : parquet_path,
            "cat_meta_path" : cat_meta_path,
            "merge_cols" : ["date"]
        }
    )
    
    # (4) Load/process/save holidays_events; track in manifest...
    print(f"Processing 'holidays_events'...")
    holidays_events = dd.read_csv(os.path.join(RAW_DATA_PATH, "holidays_events.csv"))
    holidays_events = process_holidays_events(holidays_events)
    parquet_path = os.path.join(CLEAN_DATA_PATH, "holidays_events.parquet")
    save_as_parquet(holidays_events, parquet_path)
    cat_meta_path = os.path.join(CLEAN_DATA_PATH, "holidays_events_cat_meta.json")
    save_cat_meta(holidays_events, cat_meta_path)
    manifest['secondary_data'].append(
        {
            "name" : "holidays_events",
            "parquet_path" : parquet_path,
            "cat_meta_path" : cat_meta_path,
            "merge_cols" : ["date"]
        }
    )
    
    # (5) Compute rolling stats using main + stores; track in manifest
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
    
    # Get groups
    main_stores_groups = [
        [cat_col] for cat_col in 
        list(main_stores.select_dtypes(exclude='number').columns)
        if cat_col not in ('date', 'is_train', 'is_test')
    ]
    main_stores_groups.append([])
    
    # Load main_stores into memory
    main_stores = main_stores.compute()
    
    # Calculate rolling stats wrt each group, window
    for group_cols in main_stores_groups:
        for window in windows_from_lag:
            
            # Calculate rolling stats
            start_time = time()
            print(f"Rolling stats for group '{group_cols}', window '{window}'")
            rolling_stats = compute_rolling_stats(
                main_stores, 
                cols_to_roll=['sales', 'log_sales'],
                group_cols=group_cols,
                supported_stats=supported_stats,
                quantiles=quantiles,
                lag=lag,
                window=window
            )
            
            # Save rolling stats
            suffix1 = f"_wrt_{'_'.join(group_cols)}" if group_cols else ""
            suffix2 = f"_lag{lag}_window{window}"
            file_name = f"rolling{suffix1}{suffix2}"
            parquet_path = os.path.join(CLEAN_DATA_PATH, f"{file_name}.parquet")
            save_as_parquet(rolling_stats, parquet_path)
            cat_meta_path = os.path.join(CLEAN_DATA_PATH, f"{file_name}_cat_meta.json")
            save_cat_meta(rolling_stats, cat_meta_path)
            
            # Track in manifest file
            manifest['rolling_stats'].append(
                {
                    "name" : file_name,
                    "parquet_path" : parquet_path,
                    "cat_meta_path" : cat_meta_path,
                    "merge_cols" : ["date"] + group_cols,
                }
            )
    
    # (6) Save manifest file
    manifest_path = os.path.join(CLEAN_DATA_PATH, "manifest.json")
    print(f"Saving '{manifest_path}'...")
    with open(os.path.join(CLEAN_DATA_PATH, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    client.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing Script")
    
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config.yaml",
        help="Path of config file to use"
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
