# STL imports
import os
import importlib
import types
import json

# External imports
import yaml
import pandas as pd
import dask.dataframe as dd
import pyarrow as pa
from pandas.api.types import CategoricalDtype
from tqdm import tqdm

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
            
def get_arrow_schema(ddf):
    schema = []
    for col, dtype in ddf.dtypes.items():
        if dtype.name == 'category':
            # Categorical → PyArrow dictionary type
            arrow_type = pa.dictionary(pa.int32(), pa.string())
        elif str(dtype) == 'bool':
            arrow_type = pa.bool_()
        elif str(dtype).startswith('datetime64'):
            arrow_type = pa.timestamp('ns')
        else:
            arrow_type = pa.from_numpy_dtype(dtype)
        schema.append((col, arrow_type))
    return pa.schema(schema)

def save_as_parquet(df, save_path):
    """
    Saves data as parquet
    """
    
    print(f"Saving '{save_path}'...")
    
    # Save data
    df = df.reset_index(drop=True)
    df.to_parquet(
        save_path,
        engine='pyarrow',
        schema='infer', 
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

def load_cat_meta(cat_meta_path):
    with open(cat_meta_path, 'r') as f:
        cat_meta = json.load(f)
    return cat_meta

def apply_cat_meta(ddf, cat_meta):
    # Apply categorical metadata to df
    for col, cats in cat_meta.items():
        cat_type = CategoricalDtype(
            categories=cats,
            ordered=False
        )
        ddf[col] = ddf[col].astype(cat_type)
        
    return ddf

def load_from_parquet(
    parquet_path, 
    cat_meta_path=None, 
    start_date=None,
    end_date=None,
):
    
    # Apply date filter
    filters = None
    if start_date is not None and end_date is not None:
        filters = [
            ('date', '>=', start_date), 
            ('date', '<=', end_date)
        ]
    
    # Load in parquet file
    ddf = dd.read_parquet(
        parquet_path, 
        engine="pyarrow", 
        filters=filters
    )
    
    if cat_meta_path:
        cat_meta = load_cat_meta(cat_meta_path)
        apply_cat_meta(ddf, cat_meta)

    return ddf

def load_all_cat_meta(
    cat_meta_paths
):
    # Read in categorical metadata
    all_cat_meta = {}
    for cat_meta_path in cat_meta_paths:
        cat_meta = load_cat_meta(cat_meta_path)
        for key, new_val in cat_meta.items():

            # Ensure categories are added properly
            old_val = all_cat_meta.get(key, [])
            all_cat_meta[key] = list(
                set(old_val).union(set(new_val))
            )
            
    return all_cat_meta

def load_and_merge_from_manifest(
    manifest_path, 
    sample=1.0,
    start_date=None,
    end_date=None
    ):
    
    # Keep track of cat_meta_paths (for post-merge application)
    cat_meta_paths = []
    
    # Convert start date, end date to datetime
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    
    # Open manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    # Load main table (only sample if specified)
    print("Locating 'main data' chunk...")
    main_parquet_path = manifest["main_data"]["parquet_path"]
    main_cat_meta_path = manifest["main_data"]["cat_meta_path"]
    cat_meta_paths.append(main_cat_meta_path)
    main_ddf = load_from_parquet(
        main_parquet_path, 
        main_cat_meta_path, 
        start_date=start_date,
        end_date=end_date
    )
    
    main_ddf = main_ddf.sample(frac=sample)
    
    # Iterate merging secondary data
    for meta in tqdm(manifest["secondary_data"], desc="Locating 'secondary_data' chunks..."):
        
        # Every file but 'stores' has 'date' column
        if meta['name'] == 'stores':
            secondary_ddf = load_from_parquet(
                meta["parquet_path"], 
                meta["cat_meta_path"],
                start_date=None,
                end_date=None
            )
            
        else:
            secondary_ddf = load_from_parquet(
                meta["parquet_path"], 
                meta["cat_meta_path"],
                start_date=start_date,
                end_date=end_date
            )
                
        main_ddf = main_ddf.merge(
            secondary_ddf,
            on=meta["merge_cols"],
            how="left"
        )
        
        # Save cat_meta_path
        cat_meta_paths.append(meta["cat_meta_path"])

    # Iterate merging rolling data
    for meta in tqdm(manifest["rolling_stats"], desc="Locating 'rolling_stats' chunks..."):
        rolling_ddf = load_from_parquet(
            meta["parquet_path"], 
            meta["cat_meta_path"],
            start_date=start_date,
            end_date=end_date
        )
        main_ddf = main_ddf.merge(
            rolling_ddf,
            on=meta["merge_cols"],
            how="left"
        )
        
        # Save cat_meta_path
        cat_meta_paths.append(meta["cat_meta_path"])
    
    # Apply all cat_meta (post merge)
    all_cat_meta = load_all_cat_meta(cat_meta_paths)
    main_ddf = apply_cat_meta(main_ddf, all_cat_meta)
    
    return main_ddf #.repartition(partition_size=chunksize)

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