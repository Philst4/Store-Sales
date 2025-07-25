# STL imports
import os
import importlib
import types
import json

# External imports
import yaml
import pandas as pd

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
    
def get_data_paths(storage_mode, config):
    assert storage_mode in ("local", "cloud"), f"Unidentified storage_mode '{storage_mode}'"
    raw_path = config['storage_mode'][storage_mode]['data']['raw']
    clean_path = config['storage_mode'][storage_mode]['data']['clean']
    return raw_path, clean_path

def load_raw_data(raw_path):
    print(f"Reading data from '{raw_path}'...")
    dfs = {}
    
    dfs['train'] = pd.read_csv(os.path.join(raw_path, "train.csv"))
    dfs['test'] = pd.read_csv(os.path.join(raw_path, "test.csv"))
    dfs['stores'] = pd.read_csv(os.path.join(raw_path, "stores.csv"))
    dfs['oil'] = pd.read_csv(os.path.join(raw_path, "oil.csv"))
    dfs['holidays_events'] = pd.read_csv(os.path.join(raw_path, "holidays_events.csv"))
    return dfs

def save_clean_data(clean_path, dfs):
    """
    Saves data, as well as categorical metadata.
    """
    print(f"Saving data to '{clean_path}'...")
    
    # Make path
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)
    
    # Save data
    for df_name, df in dfs.items():
        df.to_parquet(os.path.join(clean_path, f"{df_name}.parquet"), index=False)
    
        # Save categorical metadata
        cat_columns = df.select_dtypes(include='category').columns.tolist()
        cat_meta = {
            col: list(df[col].cat.categories)
            for col in cat_columns
        }
        with open(os.path.join(clean_path, f"{df_name}_cat_meta.json"), "w") as f:
            json.dump(cat_meta, f)  
    
def load_clean_data(clean_path):
    """
    Loads in clean data.
    
    Ensures that the loaded data has proper category metadata.
    """
    print(f"Loading clean data from '{clean_path}'...")
    
    # Load in data
    clean_df_names = ['main', 'stores', 'oil', 'holidays_events']
    clean_dfs = {}
    for df_name in clean_df_names:
        # Load in df
        clean_df = pd.read_parquet(
            os.path.join(
                clean_path, 
                f'{df_name}.parquet'
            ), 
            engine="pyarrow"
        )
    
        # Load in category metadata
        with open(os.path.join(clean_path, f"{df_name}_cat_meta.json"), "r") as f:
            cat_meta = json.load(f)
        
        # Assign category metadata to df
        for col, cats in cat_meta.items():
            clean_df[col] = pd.Categorical(clean_df[col], categories=cats)
            
        # Save in df dict
        clean_dfs[df_name] = clean_df
    return clean_dfs

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