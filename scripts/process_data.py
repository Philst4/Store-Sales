# STL imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# External imports
import yaml
import pandas as pd

# Internal imports
from src.data_processing import (
    combine_train_test,
    process_data,
    merge_all
)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_raw_data(raw_path):
    train = pd.read_csv(os.path.join(raw_path, "train.csv"))
    test = pd.read_csv(os.path.join(raw_path, "test.csv"))
    stores = pd.read_csv(os.path.join(raw_path, "stores.csv"))
    oil = pd.read_csv(os.path.join(raw_path, "oil.csv"))
    holidays_events = pd.read_csv(os.path.join(raw_path, "holidays_events.csv"))
    return train, test, stores, oil, holidays_events

def run_pipeline():
    config = load_config()
    RAW_DATA_PATH = config['data_paths']['raw']
    CLEAN_DATA_PATH = config['data_paths']['clean']
    
    train, test, stores, oil, holidays_events = load_raw_data(RAW_DATA_PATH)
    print(train.columns)
    print(test.columns)
    print(stores.columns)
    print(oil.columns)
    print(holidays_events.columns)
    assert False
    
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

    if not os.path.exists(CLEAN_DATA_PATH):
        os.makedirs(CLEAN_DATA_PATH)

    merged.to_csv(os.path.join(CLEAN_DATA_PATH, "clean.csv"), index=False)
    print("Processing complete.")

if __name__ == "__main__":
    run_pipeline()
