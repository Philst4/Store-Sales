import pandas as pd
import numpy as np

def calc_daily_stats(df, cols_to_roll, group_cols, supported_stats, quantiles):
    df = df.sort_values(by=group_cols + ['date'])
    
    quantile_fns = {}
    for quantile in quantiles:
        quantile_fns[quantile] = lambda x, q=quantile: np.quantile(x, float(q[1:]) / 100)

    daily_stats = (
        df
        .groupby(['date'] + group_cols, observed=True)[cols_to_roll]
        .agg(supported_stats + list(quantile_fns.values()))
        .reset_index()
        .set_index('date')
    )

    # Flatten the multi-index
    new_cols = group_cols.copy()
    for col in cols_to_roll:
        for stat in supported_stats + quantiles:
            new_cols.append(f"{col}_{stat}")
    daily_stats.columns = new_cols

    return daily_stats

def roll_daily_stats(daily_stats, group_cols, lag=16, window=1, sort=False):
    daily_stats = daily_stats.sort_values(by=group_cols).sort_index()
    
    # These are the average aggregation stats
    value_cols = [c for c in daily_stats.columns if c not in group_cols]

    if group_cols:
        rolling_stats = (
            daily_stats
            .groupby(group_cols, group_keys=False)[value_cols]
            .rolling(window=window, min_periods=1)
            .mean()
            .groupby(level=group_cols)
            .shift(lag)
            .fillna(0)
            .reset_index()
        )

    else:
        rolling_stats = (
            daily_stats[value_cols]
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(lag)
            .fillna(0)
            .reset_index()
        )
    
    return rolling_stats

def compute_rolling_stats(
        df,
        cols_to_roll, 
        group_cols, 
        supported_stats, 
        quantiles,
        lag=16, 
        window=1
    ):

    daily_stats = calc_daily_stats(
        df, 
        cols_to_roll, 
        group_cols, 
        supported_stats, 
        quantiles
    )

    rolled_stats = roll_daily_stats(daily_stats, group_cols, lag, window)
    return rolled_stats