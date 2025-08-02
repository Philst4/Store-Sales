# STL imports
import os
import importlib
import types
import json

# External imports
import yaml
import pandas as pd
import dask.dataframe as dd
from pandas.api.types import CategoricalDtype

import warnings
warnings.filterwarnings("ignore", message=".*Merging dataframes with merge column data type mismatches.*")


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
    
def get_data_paths(storage_mode, config):
    assert storage_mode in ("local", "cloud"), f"Unidentified storage_mode '{storage_mode}'"
    raw_path = config['storage_mode'][storage_mode]['data']['raw']
    clean_path = config['storage_mode'][storage_mode]['data']['clean']
    return raw_path, clean_path
            
def save_as_parquet(df, save_path):
    """
    Saves data as parquet
    """
    # Save data
    print(f"Saving '{save_path}'...")
    df = df.reset_index(drop=True)
    df.to_parquet(
        save_path,
        engine='pyarrow',
        schema='infer',  # Let pyarrow infer schema
        write_index=False,
        overwrite=True
    )
        
def save_cat_meta(df, save_path):
    # Save categorical metadata
    print(f"Saving '{save_path}'...")
    cat_columns = df.select_dtypes(include='category').columns.tolist()
    cat_meta = {
        col: list(df[col].cat.categories)
        for col in cat_columns
    }
    with open(save_path, "w") as f:
        json.dump(cat_meta, f)

def load_from_parquet(parquet_path, cat_meta_path):
    
    # Load in parquet file
    ddf = dd.read_parquet(parquet_path, engine="pyarrow")

    # Apply categorical metadata (try)
    try:
        with open(cat_meta_path, "r") as f:
            cat_meta = json.load(f)
    except:
        return ddf
    
    for col, cats in cat_meta.items():
        cat_type = CategoricalDtype(categories=cats, ordered=False)
        ddf[col] = ddf[col].astype(cat_type)

    return ddf

def load_and_merge_from_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    # Load main table
    main_parquet_path = manifest["main_data"]["parquet_path"]
    main_cat_meta_path = manifest["main_data"]["cat_meta_path"]
    main_ddf = load_from_parquet(main_parquet_path, main_cat_meta_path)
    
    # Iterate merging secondary data
    for meta in manifest["secondary_data"]:
        secondary_ddf = load_from_parquet(
            meta["parquet_path"], 
            meta["cat_meta_path"]
        )
                
        main_ddf = main_ddf.merge(
            secondary_ddf,
            on=meta["merge_cols"],
            how="left"
        )

    # Iterate merging rolling data
    for meta in manifest["rolling_stats"]:
        rolling_ddf = load_from_parquet(
            meta["parquet_path"], 
            meta["cat_meta_path"]
        )
        main_ddf = main_ddf.merge(
            rolling_ddf,
            on=meta["merge_cols"],
            how="left"
        )
    
    return main_ddf

def load_experiment_config(experiment_config_path):
    print(f"Loading experiment config from '{experiment_config_path}'...")
    
    # Load in module
    experiment_config_module = importlib.import_module(experiment_config_path)
    
    # Get module name (for filtering)
    module_name = experiment_config_module.__name__
    
    # Convert module's variables into a dictionary
    config_dict = {
        k: v for k, v in vars(experiment_config_module).items()
        if not k.startswith("__") 
        and not isinstance(v, types.ModuleType)
        and getattr(v, "__module__", module_name) == module_name
        and k != 'DEVICE'
    }
    return config_dict

def get_studies_uri(storage_mode, config):
    assert storage_mode in ("local", "cloud"), f"Unidentified storage_mode '{storage_mode}'"
    studies_uri =  config['storage_mode'][storage_mode]["studies_uri"]
    
    # Parse URI (assumes SQLite URI like 'sqlite:///./experiment_logs/optuna_studies.db')
    if storage_mode == "local":
        if studies_uri.startswith("sqlite:///"):
        
            # Make the path to the database
            path = studies_uri.replace("sqlite:///", "", 1)  # remove prefix
            dir_path = os.path.dirname(path)         # get directory path
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        else:
            raise ValueError(f"Unexpected URI format: {studies_uri}")
    else:
        raise NotImplementedError
    
    return studies_uri