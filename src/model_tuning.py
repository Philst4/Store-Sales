# STL imports
# N/A

# Internal imports
from src.modeling import (
    get_features,
    get_targets,
    build_fit_and_evaluate
)

# External imports
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import optuna
import mlflow

# THIS FILE CONTAINS EVERYTHING SPECIFIC TO TUNING

def backtest(
    df,
    build_model, 
    hyperparams,
    loss_fn,
    period_size=30,
    n_tests=12,
    n_jobs=1
):
    """
    Backtests using given arguments.
    
    Default arguments backtests using each of the 12 last months
    in the df as the validation set in each iteration (a full year). 
    
    Args:
        df : DataFrame with training data
        build_model : Function for building model
        hyperparams : Arguments for build_model
        loss_fn : How to evaluate predictions (against targets)
        period_size : Number of days in a period
        n_tests : How many tests to perform
        n_jobs : How many splits to evaluate in parallel
    
    Returns:
        losses : the list of losses accumulated for each split
    """
    
    # Sort data by date
    df = df.sort_values(by=['date'])
    last_date = df['date'].max()
    
    # Backtest
    losses = []
    print(f"#### Backtesting ####")
    for i in range(n_tests-1, -1, -1):
        print(f" * {n_tests - i} of {n_tests}...")
        
        # Figure out the dates to use
        first_val_date = last_date - pd.Timedelta(
            days=(i + 1) * period_size
        )
        last_val_date = last_date - pd.Timedelta(
            days=i * period_size
        )
        
        # Make train-val split using dates
        df_tr = df[df['date'] < first_val_date]
        df_val = df[
            (first_val_date < df['date'])
            &
            (df['date'] <= last_val_date)   
        ]
        
        if len(df_tr) == 0:
            losses.append(0.) # No training data!
        else:
            # Extract features and targets
            X_tr, y_tr = get_features(df_tr), get_targets(df_tr)
            X_val, y_val = get_features(df_val), get_targets(df_val)
            
            # Evaluate; add to losses
            losses.append(
                build_fit_and_evaluate(
                    X_tr,
                    y_tr, 
                    X_val,
                    y_val,
                    build_model,
                    hyperparams,
                    loss_fn
                )
            )
    return losses # list of losses

def make_objective(
    df,
    build_model,
    make_hyperparam_space,
    loss_fn=root_mean_squared_error, 
    experiment_name="N/A",
    loss_fn_name="RMSE",
    target_name="log_sales",
    period_size=30,
    n_tests=12,
    n_jobs=1
):
    """
    Creates an optuna objective function to use for tuning.
    
    Args:
        experiment_name : The name of the experiment
        df : The data to use to train/evaluate
        build_model : function for initializing model with hyperparams
        make_hyperparam_space : function for initializing hyperparam space
        loss_fn : evaluation function
        period_size : The backtest interval size (in days)
        n_tests : The number of backtest evaluations
        n_jobs : The number of train/evaluations to run in parallel
    """
    
    # Define function to evaluate single set of hyperparameters
    backtest_hyperparams = lambda hyperparams : backtest(
        df,
        build_model,
        hyperparams, 
        loss_fn,
        period_size,
        n_tests,
        n_jobs
    )
    
    # Define optuna objective function
    def objective(trial):
        hyperparams = make_hyperparam_space(trial) # From config file
        with mlflow.start_run(run_name=f"{experiment_name} Trial {trial.number}", nested=True):
            mlflow.set_tag("model_type", experiment_name)
            mlflow.log_param("model", experiment_name)
            mlflow.log_params(hyperparams)
            losses = backtest_hyperparams(hyperparams)
            loss = np.mean(losses)
            mlflow.log_metric(f"{loss_fn_name} of {target_name}", loss)
        return loss
    
    return objective

def run_experiment(
    objective,
    n_trials,
    study_path,
    experiment_name="N/A"
):
    """
    Runs model tuning experiment.
    
    Args:
        objective: an optuna objective to optimize over
        n_trial: number of optimization trials
        study_path: where to save the study to
        experiment_name: the name of the experiment/study
    """
    
    mlflow.set_experiment(experiment_name)
    study = optuna.create_study(
        direction='minimize',
        storage=study_path,
        study_name=experiment_name,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)
    print("\nBest trial:", study.best_trial)
    