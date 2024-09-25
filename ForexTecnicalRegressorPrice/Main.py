import os
from datetime import datetime

import pandas as pd

import Data_gathering
import Feature_engineering
import Preprocessing
import Forecasting
import Backtest
import Generate_picks
import Fees_simulator

data_gathering = False
feature_engineering = False
preprocessing = False
forecasting = True
generate_picks = True
real_trading = False

years = 10
n_assets = 500
n_weeks = 1

index = "DollarIndex"

# Define the folder path to save data
folder_name = f'data/{index}_{years}y_n{n_assets}'
os.makedirs(folder_name, exist_ok=True)
MODEL_PATH = f'stock_price_model_{index}_{n_assets}_{years}y.joblib'

if data_gathering:
    combined_df = Data_gathering.main_data_gathering(years)
    combined_df.to_parquet(f'{folder_name}/{index}_constituents_{years}y.parquet', index=False)
else:
    file_path = f'{folder_name}/{index}_constituents_{years}y.parquet'
    combined_df = pd.read_parquet(file_path)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])

if feature_engineering:
    feature_df_full = Feature_engineering.main_feature_engineering(combined_df, n_assets, n_weeks, light_mode=True)
    feature_df_full.to_parquet(f'{folder_name}/{index}_features_{years}y_n{n_assets}.parquet', index=False)
else:
    feature_df_full = pd.read_parquet(f'{folder_name}/{index}_features_{years}y_n{n_assets}.parquet')

if preprocessing:
    train_transformed, val_transformed, test_transformed, future_dataset = Preprocessing.main_preprocessing(feature_df_full)
    train_transformed.to_parquet(f'{folder_name}/train_data_{years}y_n{n_assets}.parquet')
    val_transformed.to_parquet(f'{folder_name}/val_data_{years}y_n{n_assets}.parquet')
    test_transformed.to_parquet(f'{folder_name}/test_data_{years}y_n{n_assets}.parquet')
    future_dataset.to_parquet(f'{folder_name}/future_data_{years}y_n{n_assets}.parquet')
else:
    train_transformed = pd.read_parquet(f'{folder_name}/train_data_{years}y_n{n_assets}.parquet')
    val_transformed = pd.read_parquet(f'{folder_name}/val_data_{years}y_n{n_assets}.parquet')
    test_transformed = pd.read_parquet(f'{folder_name}/test_data_{years}y_n{n_assets}.parquet')
    future_dataset = pd.read_parquet(f'{folder_name}/future_data_{years}y_n{n_assets}.parquet')

if forecasting:
    print("Forecasting...")
    forecast_data,stock_returns = Forecasting.main_forecasting(train_transformed, val_transformed, test_transformed,MODEL_PATH, MODE='train', index=index, years=years, n_assets=n_assets)
    # Save the forecast data
    forecast_data.to_parquet(f'{folder_name}/forecast_data_{years}_n{n_assets}.parquet')
    stock_returns.to_parquet(f'{folder_name}/stock_returns_{years}_n{n_assets}.parquet')
else:
    forecast_data = pd.read_parquet(f'{folder_name}/forecast_data_{years}_n{n_assets}.parquet')
    stock_returns = pd.read_parquet(f'{folder_name}/stock_returns_{years}_n{n_assets}.parquet')

if generate_picks:
    print("Generating picks...")
    Generate_picks.main_forecasting(future_dataset, n_weeks, MODEL_PATH, n_bins=5)
    #Generate_picks.main_forecasting(train_transformed, n_weeks, MODEL_PATH, n_bins=5)
    #Generate_picks.main_forecasting(val_transformed, n_weeks, MODEL_PATH, n_bins=5)

if real_trading:
    print("Real Trading...")
    investment_amount = 1000
    fees = 10 # dollar per trade
    Fees_simulator.main_forecasting(investment_amount, fees ,future_dataset, n_bins=50)
    #Fees_simulator.main_forecasting(investment_amount, fees ,train_transformed, n_bins=50)
    #Fees_simulator.main_forecasting(investment_amount, fees ,val_transformed, n_bins=50)