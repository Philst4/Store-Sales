# Internal imports


# External imports
import pandas as pd


# Just basic combination of train + test
def combine_train_test(train, test):
    # Keep track of train + test ids
    train_ids = list(train.id)
    test_ids = list(test.id)

    # Concat train + test -> main
    main = pd.concat((train.copy(), test.copy()), axis=0)
    return main, train_ids, test_ids

#### CLEANING PROCESSES ####

def clean_main(main):
    # Make store_nbr, family into categorical
    cat_features = ['store_nbr', 'family']
    main[cat_features] = main[cat_features].astype('category')

    # Convert date
    main['date'] = pd.to_datetime(main['date'], format="%Y-%m-%d")
    return main

def clean_stores(stores):
    stores = stores.copy()
    cat_cols = ['store_nbr', 'city', 'state', 'type', 'cluster']
    stores[cat_cols] = stores[cat_cols].astype('category')
    return stores

def clean_oil(oil):
    oil = oil.copy()
    oil.date = pd.to_datetime(oil.date, format="%Y-%m-%d")

    # Fill in missing dates + add day of the week
    date_range = pd.date_range(start=oil.date.min(), end=oil.date.max(), freq='D')
    dates = pd.DataFrame({'date' : date_range})
    oil = pd.merge(
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

def clean_holidays_events(holidays_events):
    holidays_events = holidays_events.copy()
    
    # Deal with transfers? 
    # Every type 'transfer' corresponds to type 'holiday'
    # Will keep the uncelebrated transfer == True rows
    # Will change the celebrated type == 'transfer' rows to holiday type
    holidays_events.loc[holidays_events['type'] == 'Transfer', 'type'] = 'Holiday'

    # Convert categorical features
    cat_features = ['type', 'locale', 'locale_name', 'description']
    holidays_events[cat_features] = holidays_events[cat_features].astype('category')

    # Convert date
    holidays_events['date'] = pd.to_datetime(holidays_events['date'], format="%Y-%m-%d")
    return holidays_events

#### FEATURE ENGINEERING PROCESSES ####

def fe_main(main):
    # Idea for future: do stuff with onpromotion (groupby's?)
    return main

def fe_holidays_events(holidays_events):
    """
    Feature engineering process for holidays_events data.
    
    Main process is creating OHE's for combos of 'locale', 'locale_name', 'type'
    """
    
    holidays_events = holidays_events.copy()
    holidays_events['combo'] = list(
        zip(
            holidays_events['locale'], 
            holidays_events['locale_name'], 
            holidays_events['type']
        )
    )
    ohes = pd.get_dummies(holidays_events['combo'], prefix='combo').astype(int)
    holidays_events = pd.concat([holidays_events['date'], ohes], axis=1)
    holidays_events = holidays_events.groupby('date').sum().reset_index()
    holidays_events['is_holiday_event'] = 1
    return holidays_events

def fe_stores(stores):
    return stores

def days_since_15th(date):
        if date.day >= 15:
            days_since = date.day - 15
        else:
            first_of_month = pd.Timestamp(year=date.year, month=date.month, day=1)
            last_day_prev_month = first_of_month - pd.offsets.Day(1)
            days_since = (last_day_prev_month.day - 15) + date.day
        return days_since

def days_since_last(date):
    # Get the last day of the current month
    first_of_month = pd.Timestamp(year=date.year, month=date.month, day=1)
    last_of_month = (first_of_month + pd.offsets.MonthEnd(1)) - pd.offsets.Day(1)
    if date == last_of_month:
        days_since = 0
    else:
        days_since = 1 + (date - first_of_month).days
    return days_since

def fe_oil(oil):
    # Mostly stuff with days
    oil = oil.sort_values(by=['date']).copy()

    # Add day of the week, week/weekend
    oil['dayOfWeek'] = oil['date'].dt.day_name().astype('category')
    oil['isWeekend'] = oil['dayOfWeek'].isin(['Saturday', 'Sunday']).astype('category')

    # Add month
    oil['Month'] = oil['date'].dt.month.astype('category')

    # Add how many days since last paycheck (from 15th/end of month)
    oil['daysSince15th'] = oil['date'].apply(days_since_15th).astype(int)
    oil['daysSinceLast'] = oil['date'].apply(days_since_last).astype(int)
    oil['daysSincePaycheck'] = oil[['daysSince15th', 'daysSinceLast']].apply(lambda x: min(x.iloc[0], x.iloc[1]), axis=1).astype(int)

    # Add oil pct change since previous days, weeks, months, etc.
    lags = [1, 2, 4, 7, 14, 28, 365]
    for lag in lags:
        oil[f'oilPrice_PctChange{lag}'] = oil['dcoilwtico'].pct_change(periods=lag).fillna(0).astype('float')
    return oil

#### Main data-processing function
def process_data(main, stores, oil, holidays_events):
    main = fe_main(clean_main(main))
    stores = fe_stores(clean_stores(stores))
    oil = fe_oil(clean_oil(oil))
    holidays_events = fe_holidays_events(
        clean_holidays_events(
            holidays_events
            )
    )
    return main, stores, oil, holidays_events

#### DATA-MERGING LOGIC ####
def fe_merge2(merge2, cols):
    """
    Some more feature-engineering on the merged dataframe.
    
    merge2 is merge between oil & holidays_events.
    """
    merge2 = merge2.copy()
    
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

def merge_all(main, stores, oil, holidays_events):
    """
    Intended to be ran after processing the raw dataframes.
    
    This may need to be reworked, because it greatly expands
    the size of the data. A normalized setup may be far better.
    """
    
    # Merge main with stores on 'store_nbr' -> merge1
    merge1 = pd.merge(main, stores, on='store_nbr', how='left')

    # Merge holiday events with oil on 'date' -> merge2
    merge2 = pd.merge(oil, holidays_events, on='date', how='left')

    # Fill N/A's in merge2
    merge2[holidays_events.columns] = merge2[holidays_events.columns].fillna(0)

    # Add info about previous/next few days being a holiday
    cols = list(holidays_events.drop(columns='date').columns) # Define columns to shift
    merge2 = fe_merge2(merge2, cols)

    # Merge merge1 with merge2 on 'date' -> merge3
    merge3 = pd.merge(merge1, merge2, on='date', how='left')
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
    merged = merged.sort_index().copy()
    merged['date'] = pd.date_range(
        start=start_date, 
        periods=len(merged),
        freq='D'
    )
    return merged
    

