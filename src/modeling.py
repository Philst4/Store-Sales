# Internal imports
import inspect

# External imports
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error

# THIS FILE CONTAINS EVERYTHING PERTAINING TO FITTING A MODEL
# Model configured to make log-predictions (predict log_sales)

# Config
NON_FEATURES = ['id', 'date', 'is_train', 'is_test', 'sales', 'log_sales']
TARGET_COL = 'log_sales'

# Basic data utils
def get_train(df):
    return df[df['is_train'] == 1]

def get_test(df):
    return df[df['is_test'] == 1]

def get_features(df):
    return df.drop(columns=NON_FEATURES)

def get_targets(df):
    return df[[TARGET_COL]]
    
def build_fit_and_evaluate(
    X_tr, 
    y_tr,
    X_val,
    y_val, 
    build_model,
    hyperparams,
    loss_fn=root_mean_squared_error
):
    """
    Fits + evaluate model on data using hyperparams.
    
    Args:
        X_tr : Training data features
        y_tr : Training data targets
        X_val : Val data features
        y_val : Val data targets
    
    Returns:
        The loss the model accumulates on the validation set
    """
    
    # Initialize model
    # Note: We want to 'encode' using all values in X_tr, X_val
    # (For the time being)
    model = build_model(
        X_tr.iloc[[0]], # Only need one sample
        hyperparams
    )
     
    # Checks if early-stopping functionality exists for model
    fit_signature = inspect.signature(model.fit)
    fit_kwargs = {}
    if "eval_set" in fit_signature.parameters and "early_stopping_rounds" in fit_signature.parameters:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["early_stopping_rounds"] = 50
        fit_kwargs["verbose"] = False

    # Fit model on training data (with appropriate kwargs)
    model.fit(X_tr, y_tr, **fit_kwargs)
    
    # Evaluate the model on test data
    y_preds = np.maximum(0, model.predict(X_val))
    
    # Debugging
    """
    assert not np.isnan(y_preds).any(), f"NaNs found in y_preds: {y_preds}"
    assert not np.isnan(y_val).any(), f"NaNs found in y_val: {y_val}"
    
    assert (y_preds >= -1).all(), "Negative values in y_preds before log1p"
    assert (y_val >= -1).all(), "Negative values in y_val before log1p"
    """
    
    return loss_fn(y_val, y_preds)
