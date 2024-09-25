import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_exchange_rate_data(ticker, start_date, end_date):
    exchange_rate = yf.Ticker(ticker)
    df = exchange_rate.history(start=start_date, end=end_date)
    return df[['Close']]

def main_data_gathering(years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    # Define the currency pairs to download from euros
    currency_pairs = {
        'USDEUR=X': 'USD/EUR',
        'GBPEUR=X': 'GBP/EUR',
        'JPYEUR=X': 'JPY/EUR',
        'CHFEUR=X': 'CHF/EUR',
        'AUDEUR=X': 'AUD/EUR',
        'CADEUR=X': 'CAD/EUR',
        'CNYEUR=X': 'CNY/EUR',
        'SEKEUR=X': 'SEK/EUR',
        'NOKEUR=X': 'NOK/EUR',
        'CZKEUR=X': 'CZK/EUR',
    }

    all_data = {}

    for ticker, name in currency_pairs.items():
        try:
            print(f"Downloading {name} exchange rate data")
            df = download_exchange_rate_data(ticker, start_date, end_date)
            all_data[name] = df['Close']
        except Exception as e:
            print(f"Error downloading {name} exchange rate data: {e}")

    if not all_data:
        print("Failed to retrieve any exchange rate data.")
        return None

    # Combine all data into a single DataFrame
    combined_df = pd.DataFrame(all_data)

    # Reset index to make date a column
    combined_df.reset_index(inplace=True)

    # Rename 'Date' column (it's named 'index' after reset_index)
    combined_df.rename(columns={'index': 'Date'}, inplace=True)

    # Convert datetime to timezone-naive
    combined_df['Date'] = combined_df['Date'].dt.tz_localize(None)

    # If there are rows with NaN values, we can fill them with the previous value
    combined_df.fillna(method='ffill', inplace=True)

    return combined_df

# Example usage
if __name__ == "__main__":
    years = 5  # You can change this to the number of years you want to retrieve data for
    exchange_rate_data = main_data_gathering(years)

    if exchange_rate_data is not None:
        print(exchange_rate_data.head())
        print(f"Data shape: {exchange_rate_data.shape}")
    else:
        print("Failed to retrieve exchange rate data.")