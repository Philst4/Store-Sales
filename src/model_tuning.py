# Internal imports
# N/A

# External imports
import pandas as pd
import numpy as np
import optuna
import mlflow

# Basic data utils
def get_train(df):
    return df[df['is_train'] == 1].copy()

def get_test(df):
    return df[df['is_test'] == 1].copy()

def get_features(df):
    return df.drop(columns=['date', 'is_train', 'is_test', 'sales']).copy()

def get_targets(df):
    return df['sales'].copy()

# More substantial
def build_fit_and_evaluate(
    X_tr, 
    y_tr,
    X_val,
    y_val, 
    build_model,
    hyperparams,
    loss_fn
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
        X_tr, 
        hyperparams
    )
    
    # Fit the model on the training data
    model.fit(X_tr, y_tr)
    
    # Evaluate the model on test data
    y_preds = np.maximum(0, model.predict(X_val))
    
    # Debugging
    assert not np.isnan(y_preds).any(), f"NaNs found in y_preds: {y_preds}"
    assert not np.isnan(y_val).any(), f"NaNs found in y_val: {y_val}"
    
    assert (y_preds >= -1).all(), "Negative values in y_preds before log1p"
    assert (y_val >= -1).all(), "Negative values in y_val before log1p"
    
    return loss_fn(y_val, y_preds)

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
            losses.append(0) # No training data!
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
    loss_fn, 
    experiment_name="N/A",
    loss_fn_name="loss",
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
            mlflow.log_metric(loss_fn_name, loss)
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
    