import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import optuna
import mlflow
from joblib import Parallel, delayed
import multiprocessing as mp
from time import time
import sys

from src.modeling import (
    get_features,
    get_targets,
    build_fit_and_evaluate
)

def _evaluate_split(df, build_model, hyperparams, loss_fn, period_size, i, last_date, counter=None, total=None, lock=None):
    """
    Helper function to evaluate one fold of the backtest.
    """
    first_val_date = last_date - pd.Timedelta(days=(i + 1) * period_size)
    last_val_date = last_date - pd.Timedelta(days=i * period_size)

    df_tr = df[df['date'] < first_val_date]
    df_val = df[(first_val_date < df['date']) & (df['date'] <= last_val_date)]

    if len(df_tr) == 0:
        result = 0.
    else:
        X_tr, y_tr = get_features(df_tr), get_targets(df_tr)
        X_val, y_val = get_features(df_val), get_targets(df_val)

        result = build_fit_and_evaluate(
            X_tr, y_tr, X_val, y_val,
            build_model, hyperparams, loss_fn
        )

    # Progress tracking
    if counter is not None and lock is not None:
        with lock:
            counter.value += 1
            print(f" * Fold {counter.value} of {total} complete", flush=True)

    return result

def _backtest(
    df,
    build_model, 
    hyperparams,
    loss_fn,
    n_backtests=4,
    valset_size=92,
    n_jobs=1
):
    """
    Backtests using given arguments, with optional parallelization.
    """
    df = df.sort_values(by=['date'])
    last_date = df['date'].max()

    print(f"#### Backtesting ({n_backtests} folds) ####")

    fold_indices = list(range(n_backtests - 1, -1, -1))

    if n_jobs == 1:
        # Serial execution
        losses = []
        for count, i in enumerate(fold_indices, start=1):
            #print(f" * Running fold {count} of {n_tests}...")
            loss = _evaluate_split(
                df, build_model, hyperparams, loss_fn,
                valset_size, i, last_date
            )
            print(f" * Fold {count} of {n_backtests} complete", flush=True)
            losses.append(loss)
    else:
        # Parallel execution
        print(f" * Running folds in parallel with n_jobs={n_jobs}...")

        manager = mp.Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()

        # Dispatch parallel jobs
        losses = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_split)(
                df, build_model, hyperparams, loss_fn,
                valset_size, i, last_date,
                counter, n_backtests, lock
            )
            for i in fold_indices
        )
        
    return losses

def make_objective(
    df,
    build_model,
    make_hyperparam_space,
    loss_fn=root_mean_squared_error, 
    experiment_name="N/A",
    loss_fn_name="RMSE",
    target_name="log_sales",
    n_backtests=4,
    valset_size=92,
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
    backtest_hyperparams = lambda hyperparams : _backtest(
        df,
        build_model,
        hyperparams, 
        loss_fn,
        n_backtests,
        valset_size,
        n_jobs
    )
    
    # Define optuna objective function
    def objective(trial):
        
        # From experiment config file
        hyperparams = make_hyperparam_space(trial)
        with mlflow.start_run(run_name=f"{experiment_name} Trial {trial.number}", nested=True):
            mlflow.set_tag("model_type", experiment_name)
            mlflow.log_param("model", experiment_name)
            mlflow.log_params(hyperparams)
            losses = backtest_hyperparams(hyperparams)
            loss = np.mean(losses)
            mlflow.log_metric(f"{loss_fn_name} of {target_name}", loss)
        return loss
    
    return objective

#### NEW ####

def _run_backtest_worker(df, build_model, loss_fn, n_backtests, valset_size, n_jobs, hyperparams, out_q):
    """
    Top-level worker so it's picklable under 'spawn'. Computes loss and
    sends a dict back via a Queue.
    """
    try:
        losses = _backtest(
            df,
            build_model,
            hyperparams,
            loss_fn,
            n_backtests,
            valset_size,
            n_jobs
        )
        out_q.put({"ok": True, "loss": float(np.mean(losses))})
    except Exception as e:
        # Send the error back so parent can log why it pruned
        out_q.put({"ok": False, "error": repr(e)})

def make_objective(
    df,
    build_model,
    make_hyperparam_space,
    loss_fn=root_mean_squared_error, 
    experiment_name="N/A",
    loss_fn_name="RMSE",
    target_name="log_sales",
    n_backtests=4,
    valset_size=92,
    n_jobs=1,
    max_time_per_trial=300
):
    def objective(trial):
        hyperparams = make_hyperparam_space(trial)

        with mlflow.start_run(run_name=f"{experiment_name} Trial {trial.number}", nested=True):
            mlflow.set_tag("model_type", experiment_name)
            mlflow.log_param("model", experiment_name)
            mlflow.log_params(hyperparams)

            q = mp.Queue()
            p = mp.Process(
                target=_run_backtest_worker,
                args=(df, build_model, loss_fn, n_backtests, valset_size, n_jobs, hyperparams, q),
            )
            p.start()

            try:
                # Wait for a result from the worker, up to max_time_per_trial
                result = q.get(timeout=max_time_per_trial)
            except Exception:
                # Timed out or queue error
                if p.is_alive():
                    p.terminate()
                p.join()
                mlflow.log_param("pruned_reason", "timeout")
                raise optuna.TrialPruned()

            # Ensure process is cleaned up
            p.join()

            # Log exit code for extra visibility
            try:
                mlflow.log_param("mp_exitcode", p.exitcode)
            except Exception:
                pass

            if result.get("ok"):
                loss = result["loss"]
                mlflow.log_metric(f"{loss_fn_name} of {target_name}", loss)
                return loss
            else:
                mlflow.log_param("pruned_reason", f"backtest_failed: {result.get('error', 'unknown')}")
                raise optuna.TrialPruned()

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
    