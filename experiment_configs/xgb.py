# Internal imports
import warnings

# External imports
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from xgboost import XGBRegressor
import torch # for device checking

# Definitions
experiment_name = "xgb"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Check for GPU

def build_model(X_sample, hyperparams):
    # Identify categorical columns
    cat_cols = [col for col in X_sample.columns if X_sample[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_sample[col])]

    # Identify numeric columns
    num_cols = [col for col in X_sample.columns if pd.api.types.is_numeric_dtype(X_sample[col])]

    # Define transformer for categorical columns
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Build ColumnTransformer
    # What this does, is handles each column's type as specified.
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_cols),
            ('num', 'passthrough', num_cols)
        ],
        remainder='drop'  # drop any columns not listed in transformers
    )

    # Wrap in pipeline
    # What this does, is preprocesses the data, then fits the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(**hyperparams))
    ])
    return model

def add_constant_hyperparams(hyperparams):
    constant_hyperparams = {
        'seed' : 42,
        'objective' : 'reg:squarederror',
        'eval_metric' : 'rmse',
        'tree_method' : 'hist',
        'enable_categorical' : True,
        'device' : DEVICE,
        'max_bin' : 256,
        'single_precision_histogram' : True,
    }
    return hyperparams | constant_hyperparams

def make_hyperparam_space(trial):
    # Hyperparameter suggestions
    hyperparam_space =  {
        'n_estimators' : trial.suggest_int('n_estimators', 50, 500),
        'max_depth' : trial.suggest_int('max_depth', 2, 10),
        'learning_rate' : trial.suggest_float('learning_rate', 1e-3, 1e0, log=True),
        'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda' : trial.suggest_float('reg_lambda', 0, 10),
        'gamma' : trial.suggest_float('gamma', 0, 5),    
    }
    return add_constant_hyperparams(hyperparam_space)