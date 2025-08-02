# Internal imports

# External imports
import pandas as pd
import numpy as np
import dask.dataframe as dd
import logging

logging.getLogger('distributed.shuffle').setLevel(logging.ERROR)

# Just basic combination of train + test
def combine_train_test(train, test):
    # Keep track of train + test ids
    train_ids = train['id'].compute().tolist()
    test_ids = test['id'].compute().to_list()

    # Concat train + test -> main
    main = dd.concat([train, test], axis=0)
    main = main.assign(
        is_train=main['id'].isin(train_ids).astype(int),
        is_test=main['id'].isin(test_ids).astype(int)
    )
    return main, train_ids, test_ids

#### CLEANING PROCESSES ####

def _clean_main(main):
    # Make store_nbr, family into categorical
    cat_features = ['store_nbr', 'family']
    main = main.categorize(columns=cat_features)

    # Convert date
    main['date'] = dd.to_datetime(main['date'], format="%Y-%m-%d")
    return main

def _clean_stores(stores):
    cat_cols = ['store_nbr', 'city', 'state', 'type', 'cluster']
    stores = stores.categorize(columns=cat_cols)
    return stores

def _clean_oil(oil):
    oil.date = dd.to_datetime(oil.date, format="%Y-%m-%d")

    # Fill in missing dates + add day of the week
    # Fill missing dates (requires compute for date range)
    min_date, max_date = oil['date'].min().compute(), oil['date'].max().compute()
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    dates = dd.from_pandas(pd.DataFrame({'date': date_range}), npartitions=1)
    oil = dd.merge(
        oil, 
        dates,
        on='date',
        how='right'
    )

    # Fill in missing oil values
    oil = oil.sort_values(by=['date'])
    oil['dcoilwtico'] = oil['dcoilwtico'].ffill()
    
    # Just fill in the first N/A value with the second value
    oil['dcoilwtico'] = oil['dcoilwtico'].bfill()
    return oil

def _clean_holidays_events(holidays_events):
    # Deal with transfers? 
    # Every type 'transfer' corresponds to type 'holiday'
    # Will keep the uncelebrated transfer == True rows
    # Will change the celebrated type == 'transfer' rows to holiday type
    holidays_events['type'] = holidays_events['type'].mask(
        holidays_events['type'] == 'Transfer', 'Holiday')
    
    # Convert categoricals
    cat_features = ['type', 'locale', 'locale_name', 'description']
    holidays_events = holidays_events.categorize(columns=cat_features)
    
    # Convert date
    holidays_events['date'] = dd.to_datetime(holidays_events['date'], format="%Y-%m-%d")
    return holidays_events

#### FEATURE ENGINEERING PROCESSES ####
def _fe_main(
    main
):
    # Add log2p of 'sales' (target)
    main['log_sales'] = np.log1p(main['sales'])
    return main

def _fe_holidays_events(holidays_events):
    """
    Feature engineering process for holidays_events data.
    
    Main process is creating OHE's for combos of 'locale', 'locale_name', 'type'
    """
    holidays_events = holidays_events.compute()
    
    holidays_events['combo'] = list(
        zip(
            holidays_events['locale'], 
            holidays_events['locale_name'], 
            holidays_events['type']
        )
    )
    ohes = pd.get_dummies(holidays_events['combo'], prefix='combo')
    ohe_cols = list(ohes.columns)
    holidays_events = pd.concat([holidays_events['date'], ohes], axis=1)
    holidays_events = holidays_events.groupby('date').sum().reset_index()
    holidays_events[ohe_cols] = holidays_events[ohe_cols].clip(lower=0, upper=1).astype('category')
    holidays_events['is_holiday_event'] = 1
    return dd.from_pandas(holidays_events, npartitions=1)

def _fe_stores(stores):
    return stores

def _days_since_15th(date):
        if date.day >= 15:
            days_since = date.day - 15
        else:
            first_of_month = pd.Timestamp(year=date.year, month=date.month, day=1)
            last_day_prev_month = first_of_month - pd.offsets.Day(1)
            days_since = (last_day_prev_month.day - 15) + date.day
        return days_since

def _days_since_last(date):
    # Get the last day of the current month
    first_of_month = pd.Timestamp(year=date.year, month=date.month, day=1)
    last_of_month = (first_of_month + pd.offsets.MonthEnd(1)) - pd.offsets.Day(1)
    if date == last_of_month:
        days_since = 0
    else:
        days_since = 1 + (date - first_of_month).days
    return days_since

def _fe_oil(oil, windows_from_0, lag, windows_from_lag):
    # Convert to dataframe
    oil = oil.compute()
    
    # Mostly stuff with days
    oil = oil.sort_values(by=['date'])

    # Add day of the week, week/weekend
    oil['dayOfWeek'] = oil['date'].dt.day_name().astype('category')
    oil['isWeekend'] = oil['dayOfWeek'].isin(['Saturday', 'Sunday']).astype('category')

    # Add month
    oil['Month'] = oil['date'].dt.month.astype('category')

    # Add how many days since last paycheck (from 15th/end of month)
    oil['daysSince15th'] = oil['date'].apply(_days_since_15th).astype(int)
    oil['daysSinceLast'] = oil['date'].apply(_days_since_last).astype(int)
    oil['daysSincePaycheck'] = oil[['daysSince15th', 'daysSinceLast']].apply(lambda x: min(x.iloc[0], x.iloc[1]), axis=1).astype(int)

    # Add oil pct change since previous days, weeks, months, etc.
    # From lag=0
    windows_from_0 = [1, 2, 4, 7, 14]
    for window in windows_from_0:
        oil[f'oilPrice_PctChange_lag0_window{window}'] = (
            oil['dcoilwtico']
            .pct_change(periods=window)
            .fillna(0)
            .astype('float')
        )
        
    # From lag=lag
    lag = 16
    windows_from_lag = [1, 7, 14, 28, 91, 365]
    for window in windows_from_lag:
        oil[f'oilPrice_PctChange_lag{lag}_window{window}'] = (
            oil['dcoilwtico']
            .pct_change(periods=lag+window)
            .fillna(0)
            .astype('float')
        )
    return dd.from_pandas(oil)

def process_main(main):
    return _fe_main(_clean_main(main))

def process_stores(stores):
    return _fe_stores(_clean_stores(stores))

def process_oil(
    oil,
    windows_from_0=[1, 2, 4, 7, 14],
    lag=16,
    windows_from_lag=[1, 7, 28, 91, 365]
):
    return _fe_oil(
        _clean_oil(oil), 
        windows_from_0, 
        lag, 
        windows_from_lag
    )

def process_holidays_events(holidays_events):
    return _fe_holidays_events(_clean_holidays_events(holidays_events))

def _compute_rolling_stats(
    main, 
    group_cols, 
    col, 
    lag, 
    window, 
    quantiles=[], 
    show_progress=False
):
    """
    Compute lagged rolling stats on `col` in `main`, grouped by `group_cols`.
    If group_cols is None or empty, computes globally.
    """
    # Basic statistics we always compute
    stats = ['mean', 'std', 'min', 'max']
    
    # Add quantile calculations if specified
    if quantiles:
        # Convert quantiles to fractions (e.g., 90 -> 0.9)
        quantile_stats = {f'q{q}': (q/100.0) for q in quantiles}
    else:
        quantile_stats = {}

    # If no group, treat whole DataFrame
    if not group_cols:
        daily_sum = (
            main
            .groupby('date')[col]
            .sum()
            .reset_index()
            .set_index('date')
        )
        
        # Compute basic stats
        rolled = daily_sum[[col]].rolling(f"{window}D", min_periods=1).agg(stats)
        
        # Compute quantiles if specified
        if quantiles:
            for q_name, q_value in quantile_stats.items():
                rolled[q_name] = daily_sum[col].rolling(f"{window}D", min_periods=1).quantile(q_value)

    # Grouped version
    else:
        daily_sum = (
            main
            .groupby(['date'] + group_cols, observed=False)[col]
            .sum()
            .reset_index()
            .set_index('date')
        )
        
        # Compute basic stats
        rolled = (
            daily_sum
            .groupby(group_cols, observed=False)[col]
            .rolling(f"{window}D", min_periods=1)
            .agg(stats)
        )
        
        # Compute quantiles if specified
        if quantiles:
            for q_name, q_value in quantile_stats.items():
                rolled[q_name] = (
                    daily_sum
                    .groupby(group_cols, observed=False)[col]
                    .rolling(f"{window}D", min_periods=1)
                    .quantile(q_value)
                )

    # Rename columns
    suffix = f"_wrt_{'_'.join(group_cols)}"
    new_columns = []
    for stat in stats:
        new_columns.append(f"{col}_lag{lag}_window{window}_{stat}{suffix}")
    
    for q_name in quantile_stats.keys():
        q_num = q_name[1:]  # Remove 'q' prefix
        new_columns.append(f"{col}_lag{lag}_window{window}_q{q_num}{suffix}")
    
    rolled.columns = new_columns
    
    # Reset index
    rolled = rolled.reset_index()
    
    # Shift dates back by 'lag'
    rolled['date'] = rolled['date'] + pd.DateOffset(days=lag)
    return rolled



def compute_rolling_stats(
    main, 
    group_cols, 
    rolling_cols, 
    lag, 
    window, 
    quantiles=[], 
    suffix="",
    show_progress=False
):
    """
    Compute lagged rolling stats on columns in `rolling_cols` in `main`, grouped by `group_cols`.
    If group_cols is None or empty, computes globally.
    
    Assumes 'date' is index.
    """
    
    # need to be a pd.dataframe for now
    if isinstance(main, dd.DataFrame):
        main = main.compute()
    
    # Basic statistics we always compute
    stats = ['mean', 'std', 'min', 'max']
    
    # Add quantile calculations if specified
    if quantiles:
        # Convert quantiles to fractions (e.g., 90 -> 0.9)
        quantile_stats = {f'q{q}': (q/100.0) for q in quantiles}
    else:
        quantile_stats = {}

    # If no group, treat whole DataFrame
    if not group_cols:
        daily_sum = (
            main
            .groupby('date')[rolling_cols]
            .sum()
            .reset_index()
            .set_index('date')
        )
        
        # Compute basic stats for all columns at once
        rolled = daily_sum[rolling_cols].rolling(f"{window}D", min_periods=1).agg(stats)
        
        # Compute quantiles if specified
        if quantiles:
            for q_name, q_value in quantile_stats.items():
                for col in rolling_cols:
                    rolled["_".join([col, q_name])] = daily_sum[col].rolling(f"{window}D", min_periods=1).quantile(q_value)

    # Grouped version
    else:
        daily_sum = (
            main
            .groupby(['date'] + group_cols, observed=False)[rolling_cols]
            .sum()
            .reset_index()
            .set_index('date')
        )
        
        # Compute basic stats for all columns at once
        rolled = (
            daily_sum
            .groupby(group_cols, observed=False)[rolling_cols]
            .rolling(f"{window}D", min_periods=1)
            .agg(stats)
        )
        
        # Compute quantiles if specified
        if quantiles:
            for q_name, q_value in quantile_stats.items():
                for col in rolling_cols:
                    rolled["_".join([col, q_name])] = (
                        daily_sum
                        .groupby(group_cols, observed=False)[col]
                        .rolling(f"{window}D", min_periods=1)
                        .quantile(q_value)
                    )

    # Flatten MultiIndex columns and create proper names
    new_columns = []
    
    # Handle basic stats
    for col in rolling_cols:
        for stat in stats:
            new_columns.append(f"{col}_lag{lag}_window{window}_{stat}{suffix}")
    
    # Handle quantiles
    if quantiles:
        for col in rolling_cols:
            for q_name in quantile_stats.keys():
                q_num = q_name[1:]  # Remove 'q' prefix
                new_columns.append(f"{col}_lag{lag}_window{window}_q{q_num}{suffix}")
    
    # Reset index, rename columns
    rolled.columns = ['_'.join(col).strip() for col in rolled.columns]
    rolled.columns = new_columns
    rolled = rolled.reset_index()

    # Shift dates back by 'lag'
    rolled['date'] = pd.to_datetime(rolled['date'], format="%Y-%m-%d")
    rolled['date'] = rolled['date'] + pd.DateOffset(days=lag)
    
    return dd.from_pandas(rolled, npartitions=1)

def _get_lag_stats(
    main, 
    lag=16, 
    windows=[1, 7, 14, 28, 91, 365], 
    cols=['sales'], 
    groups=[],
    quantiles=[5, 25, 50, 75, 95]
):
    """
    Efficiently computes lagged rolling statistics for `col` across different groupings.
    Each stat for row at date D is computed over range (D - lag - window, D - lag].
    """
    assert 'date' in main.columns, "'date' not a column in the df"
    main = main.sort_values('date')
    
    # Set 'date' as index
    main.set_index('date', inplace=True)

    # For keeping track of newly rolled dfs + how to merge them onto main    
    rolls = [] # List of dataframes
    merge_cols = [] # List of list of columns
    
    for col in cols:
        for window in windows:
            # 1. All Stores, All Families
            rolls.append(
                _compute_rolling_stats(
                    main, 
                    group_cols=[], 
                    col=col, 
                    lag=lag, 
                    window=window, 
                    suffix='',
                    quantiles=quantiles
                )
            )
            merge_cols.append(['date'])
            
            # 2. wrt cat_cols
            for group in groups:
                suffix = f"_wrt_{'_'.join(group)}"
                rolled = _compute_rolling_stats(
                    main, 
                    group_cols=group, 
                    col=col, 
                    lag=lag, 
                    window=window, 
                    suffix=suffix,
                    quantiles=quantiles
                )
                
                rolls.append(rolled)
                merge_cols.append(['date'] + group)
    
    return rolls, merge_cols # list of dataframes, list of columns to merge


#### Main data-processing function
def process_data(
    main, 
    stores, 
    oil, 
    holidays_events,
    windows_from_0=[1, 2, 4, 7, 14],
    lag=16,
    windows_from_lag=[1, 7, 14, 28, 91, 365], # 13 weeks is 91 days
    quantiles=[0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
):
    main = _fe_main(_clean_main(main))
    stores = _fe_stores(_clean_stores(stores))
    oil = _fe_oil(
        _clean_oil(oil),
        windows_from_0,
        lag,
        windows_from_lag
    )
    holidays_events = _fe_holidays_events(
        _clean_holidays_events(
            holidays_events
            )
    )
    
    # Merge main and stores on store_nbr, get groups    
    main_stores = pd.merge(
        main,
        stores,
        on='store_nbr',
        how='left',
    )
    main_stores['store_nbr'] = main_stores['store_nbr'].astype('category')
    
    main_stores_groups = [
        [cat_col] for cat_col in 
        list(main_stores.select_dtypes(exclude='number').columns)
        if cat_col != 'date'
    ]
    
    # Get lag stats for 'main_stores'
    main_stores_lag_stats, main_stores_merge_cols = _get_lag_stats(
        main_stores, 
        lag, 
        windows_from_lag,
        cols=['sales', 'log_sales'],
        groups=main_stores_groups,
        quantiles=quantiles
    )
    
    # Merge 'main_stores_lag_stats' with with 'main_stores'
    cols_to_fill = []
    for i in range(len(main_stores_lag_stats)):
        rolling_df = main_stores_lag_stats[i]
        merge_cols = main_stores_merge_cols[i]
        
        main_stores = pd.merge(
            main_stores,
            rolling_df,
            on=merge_cols,
            how='left'
        )
        
        # Explicitly convert merge cols back to categorical columns
        cat_cols = [col for col in merge_cols if col != 'date']
        main_stores[cat_cols] = main_stores[cat_cols].astype('category')
        
        # Keep track of 'fill' cols
        cols_to_fill += list(main_stores_lag_stats[i].select_dtypes(include='number').columns)
        
    # Fill N/A's of new columns with 0
    main_stores[cols_to_fill] = main_stores[cols_to_fill].fillna(0.)
      
    # Add feature to show how much run-way rolled stats get for each row
    # Just how many days of previous data were available
    min_date = main_stores['date'].min()
    main_stores = main_stores.assign(
        n_prev_days=(main_stores['date'] - min_date).dt.days.clip(upper=365)
    )
    return main_stores, stores, oil, holidays_events

#### DATA-MERGING LOGIC ####
def _fe_merge2(merge2, cols):
    """
    Some more feature-engineering on the merged dataframe.
    
    merge2 is merge between oil & holidays_events.
    """
    # Add info about previous/next few days being a holiday

    # Step 2: Create all shifted DataFrames
    shifted_1next = merge2[cols].shift(-1).fillna(0).astype(int)
    shifted_1next.columns = [col + '_1next' for col in cols]

    shifted_2next = merge2[cols].shift(-2).fillna(0).astype(int)
    shifted_2next.columns = [col + '_2next' for col in cols]

    shifted_1before = merge2[cols].shift(1).fillna(0).astype(int)
    shifted_1before.columns = [col + '_1before' for col in cols]

    shifted_2before = merge2[cols].shift(2).fillna(0).astype(int)
    shifted_2before.columns = [col + '_2before' for col in cols]

    merge2 = pd.concat([merge2, shifted_1next, shifted_2next, shifted_1before, shifted_2before], axis=1)
    return merge2

def _fe_merge2(df, cols):
    """
    Adds shifted versions of holiday/oil indicators (before/after) to the merged DataFrame.
    Compatible with both pandas and Dask.
    """
    is_dask = isinstance(df, dd.DataFrame)
    concat_func = dd.concat if is_dask else pd.concat

    # Ensure sorted by date before shifting
    df = df.sort_values('date')

    def shift_cols(df, shift_val, suffix):
        shifted = df[cols].shift(shift_val).fillna(0).astype(int)
        shifted.columns = [f"{col}_{suffix}" for col in cols]
        return shifted

    shifted_1next = shift_cols(df, -1, "1next")
    shifted_2next = shift_cols(df, -2, "2next")
    shifted_1before = shift_cols(df, 1, "1before")
    shifted_2before = shift_cols(df, 2, "2before")

    # Combine all
    df = concat_func([df, shifted_1next, shifted_2next, shifted_1before, shifted_2before], axis=1)

    return df

def merge_all(main, stores, oil, holidays_events):
    """
    Merges all input DataFrames. Works for both pandas and Dask DataFrames.
    
    Assumes:
    - main and stores are merged on 'store_nbr'
    - oil and holidays_events are merged on 'date'
    - final merge is on 'date'
    """

    is_dask = isinstance(main, dd.DataFrame)

    print(f"Merging all data using {'Dask' if is_dask else 'Pandas'}...")

    merge_func = dd.merge if is_dask else pd.merge

    # Need to make 'store_nbr' categories align
    if is_dask:
        
        # Dask-compatible categorical alignment
        main_cats = main['store_nbr'].drop_duplicates().compute().cat.categories
        store_cats = stores['store_nbr'].drop_duplicates().compute().cat.categories        
        all_cats = main_cats.union(store_cats)
        
        # Assign all categories to each side
        main['store_nbr'] = main['store_nbr'].cat.set_categories(all_cats)
        stores['store_nbr'] = stores['store_nbr'].cat.set_categories(all_cats)
        
    merge1 = merge_func(main, stores, on='store_nbr', how='left')

    # Merge oil with holidays_events on 'date'
    merge2 = merge_func(oil, holidays_events, on='date', how='left')

    # Fill NA only in holidays_events columns
    fill_cols = holidays_events.columns.difference(['date'])
    for col in fill_cols:
        merge2[col] = merge2[col].fillna(0)
    #merge2[fill_cols] = merge2[fill_cols].fillna(0)

    # Shift holiday columns (this must be compatible with Dask)
    merge2 = _fe_merge2(merge2, fill_cols)

    # Final merge on 'date'
    merge3 = merge_func(merge1, merge2, on='date', how='left')

    return merge3


def assign_ascending_dates(merged):
    """
    Function for assigning ascending dates to the dataframe.
    
    Starts from the min date in the dataframe. This function is
    for diversifying the dates in the dataframe for test modes, 
    in order to ensure that the program will run as expected in 
    production.
    
    Args:
    * merged (pd.DataFrame) : The merged dataframe, contains 'date' column
    
    Returns:
    * merged (pd.DataFrame) : The dataframe with ascending dates
    """

    # Get min date
    start_date = merged['date'].min()
    
    # Assign ascending dates
    merged = merged.sort_index()
    merged['date'] = pd.date_range(
        start=start_date, 
        periods=len(merged),
        freq='D'
    )
    return merged
    

