import pandas as pd
import numpy as np
import re

#read SP500_historical_financials.parquet"



def split_data(df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.3, time_column='Date'):

    # Ensure ratios sum to 1
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)

    # Sort the dataframe by date
    df = df.sort_values(time_column)

    # Get unique dates
    unique_dates = df[time_column].unique()
    n_dates = len(unique_dates)

    # Calculate split indices
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))

    # Split dates
    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    # Split dataframe
    train_df = df[df[time_column].isin(train_dates)]
    val_df = df[df[time_column].isin(val_dates)]
    test_df = df[df[time_column].isin(test_dates)]

    return train_df, val_df, test_df


def transform_dataset(data, create_future_price):
    print("Transforming dataset...Number of rows before transformation: ", data.shape[0])
    # Melt the dataframe
    melted_df = pd.melt(data,
                        id_vars=['Date'],
                        var_name='Stock_Feature',
                        value_name='Value')

    # Split the Stock_Feature column into Stock and Feature
    print("Splitting Stock_Feature column...")
    melted_df[['Stock', 'Feature']] = melted_df['Stock_Feature'].str.split('_', n=1, expand=True)

    # Pivot the dataframe to get features as separate columns
    print("Pivoting the dataframe...")
    output_df = melted_df.pivot_table(index=['Date', 'Stock'], columns='Feature', values='Value').reset_index()

    # Reset column names
    output_df.columns.name = None

    # Reorder columns
    print("Reordering columns...")
    column_order = ['Date', 'Stock', 'Price', "Target"] + [col for col in output_df.columns if col not in ['Date', 'Stock', 'Price', "Target"]]
    output_df = output_df[column_order]

    #If Price_Futuro_1W is NaN, drop the row
    output_future = output_df.copy()
    output_df = output_df.dropna(subset=['Target'])

    if create_future_price:
        return  output_future

    return output_df


def main_preprocessing(data):

    # Fill NaN or Inf values with the average nearby values
    data = data.replace([np.inf, -np.inf], np.nan)
    train_data, val_data, test_data = split_data(data)

    train_transformed = transform_dataset(train_data, False)
    val_transformed = transform_dataset(val_data,False)
    test_transformed = transform_dataset(test_data,False)
    future_dataset = transform_dataset(test_data,True)

    return train_transformed, val_transformed, test_transformed, future_dataset

