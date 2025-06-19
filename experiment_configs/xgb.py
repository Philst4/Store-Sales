import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np

# Definitions
experiment_name = "xgb"

def build_model(df, hyperparams):
    # Identify categorical columns
    cat_cols = [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]

    # Identify numeric columns
    num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

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
    
def loss_fn(y_true, y_pred):
    """
    RMLSE implementation.
    """
    return root_mean_squared_error(np.log1p(y_true), np.log1p(y_pred))
    
loss_fn_name = "RMLSE"
period_size = 30
n_tests = 12