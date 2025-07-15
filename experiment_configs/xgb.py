import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np

# Definitions
experiment_name = "xgb"

# Check for GPU
def gpu_is_available():
    try:
        params = {
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
            "n_estimators": 1
        }
        dtrain = xgb.DMatrix(data=[[0, 0], [1, 1]], label=[0, 1])
        model = xgb.train(params, dtrain, num_boost_round=1)
        return True
    except xgb.core.XGBoostError:
        return False
    
USE_GPU = gpu_is_available()

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
    
    # Build predictor
    if USE_GPU:
        hyperparams['tree_method'] = "gpu_hist"
        hyperparams['predictor'] = "gpu_predictor"
    else:
        hyperparams['tree_method'] = "hist"
        hyperparams['predictor'] = "auto"   

    # Wrap in pipeline
    # What this does, is preprocesses the data, then fits the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(**hyperparams))
    ])
    return model

def make_hyperparam_space(trial):
    # Hyperparameter suggestions
    return {
        'seed' : 42,
        'objective' : 'reg:squarederror',
        'eval_metric' : 'rmse',
        'n_estimators' : trial.suggest_int('n_estimators', 50, 500),
        'max_depth' : trial.suggest_int('max_depth', 2, 10),
        'learning_rate' : trial.suggest_float('learning_rate', 1e-3, 1e0, log=True),
        'subsample' : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda' : trial.suggest_float('reg_lambda', 0, 10),
        'gamma' : trial.suggest_float('gamma', 0, 5),    
    }

period_size = 30
n_tests = 12