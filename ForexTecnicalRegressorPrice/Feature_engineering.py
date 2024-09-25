import numpy as np
import pandas as pd
from scipy import stats
import talib

def create_stock_features_and_target(df,n_weekds, test_mode=False, test_stock=None):
    # Ensure the dataframe is sorted by date
    df = df.sort_values('Date')

    # List to store all new features and targets
    new_features = []

    # Determine which columns to process
    if test_mode and test_stock:
        if test_stock not in df.columns:
            raise ValueError(f"Test stock {test_stock} not found in the dataframe")
        columns_to_process = [test_stock]
    else:
        columns_to_process = [col for col in df.columns if col != 'Date']

    for column in columns_to_process:
        # Create features for each stock
        print(f"Processing stock: {column}")
        open = df[column]

        # Calculate returns
        daily_return = open.pct_change()

        fibonacci_period = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

        # Moving averages
        ma_windows = fibonacci_period
        ma_features = {f'MA_{window}': open.rolling(window=window).mean() for window in ma_windows}
        ma_ratio_features = {f'MA_{window}_Ratio': open / ma_features[f'MA_{window}'] for window in ma_windows}

        # Volatility
        volatility_windows = fibonacci_period
        volatility_features = {f'Volatility_{window}': daily_return.rolling(window=window).std() for window in volatility_windows}

        # Momentum
        momentum_windows = fibonacci_period
        momentum_features = {f'Momentum_{window}': open.pct_change(periods=window) for window in momentum_windows}

        # Price lags with fibonacci period
        price_lags = fibonacci_period
        price_lags_features = {f'Price_Lag_{window}': open.shift(window) for window in price_lags}

        # Relative Strength Index (RSI)
        rsi = talib.RSI(open)

        # MACD
        macd, macd_signal, _ = talib.MACD(open)

        # Bollinger Bands
        upper_band, middle_band, lower_band = talib.BBANDS(open)
        bb_width = (upper_band - lower_band) / middle_band


        # Rate of Change (ROC)
        roc = talib.ROC(open)


        # Exponential Moving Average (EMA)
        ema = talib.EMA(open)

        # Kurtosis
        kurtosis = daily_return.rolling(window=30).apply(lambda x: stats.kurtosis(x))

        Target = open.shift(-n_weekds*5)

        stock_features = {
            **{f'{column}_{k}': v for k, v in ma_features.items()},
            **{f'{column}_{k}': v for k, v in ma_ratio_features.items()},
            **{f'{column}_{k}': v for k, v in volatility_features.items()},
            **{f'{column}_{k}': v for k, v in momentum_features.items()},
            **{f'{column}_{k}': v for k, v in price_lags_features.items()},
            f'{column}_RSI': rsi,
            f'{column}_MACD': macd,
            f'{column}_MACD_Signal': macd_signal,
            f'{column}_Price': open,
            f'{column}_BB_Upper': upper_band,
            f'{column}_BB_Lower': lower_band,
            f'{column}_BB_Width': bb_width,
            f'{column}_ROC': roc,
            f'{column}_EMA': ema,
            f'{column}_Kurtosis': kurtosis,
            f'{column}_Target': Target,
        }

        new_features.append(pd.DataFrame(stock_features))

    # Combine all new features with the original dataframe
    if test_mode:
        result_df = pd.concat([df[['Date', test_stock]]] + new_features, axis=1)
    else:
        result_df = pd.concat([df] + new_features, axis=1)

    return result_df


def main_feature_engineering(df, n_stocks,n_weekds, light_mode):

    if light_mode:
        df = df.iloc[:, :n_stocks]

    feature_df_full = create_stock_features_and_target(df,n_weekds)

    return feature_df_full


#I wat to select only the nice functions with selectkbest



