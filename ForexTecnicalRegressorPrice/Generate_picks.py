import pandas as pd
import numpy as np
import joblib
import optuna
import warnings
from  datetime import datetime
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

def prepare_data(test):
    target_column = 'Target'  # Assuming 'Price' is the target for regression

    feature_columns = [col for col in test.columns if
                       col not in ['Date', 'Stock', target_column, 'Target', "Price"]]

    X_test = test[feature_columns]
    test_date = test[['Date', 'Stock', 'Price']]

    return X_test, test_date

def run_model(model, X_test):
    test_predictions = model.predict(X_test)
    return test_predictions, X_test

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def main_forecasting(test,n_weeks,MODEL_PATH,n_bins):
    forecasting_horizon = n_weeks*7
    X_test, test_date = prepare_data(test)

    model = load_model(MODEL_PATH)

    test_predictions, X_test = run_model(model, X_test)

    # Re-combine the test_date and predictions
    test_predictions = pd.DataFrame(test_predictions, columns=['Price_Prediction'])
    test_predictions = pd.concat([test_date, test_predictions], axis=1)

    stock_returns = test_predictions[["Date", "Stock", "Price"]]
    test_predictions = test_predictions.pivot(index='Date', columns='Stock', values='Price_Prediction')
    stock_returns = stock_returns.pivot(index='Date', columns='Stock', values='Price')

    test_predictions = test_predictions.pct_change().dropna()
    stock_returns = stock_returns.pct_change().dropna()

    forecast_data = test_predictions.reset_index().melt(id_vars='Date', var_name='Stock', value_name='Price_Prediction')

    # Create 20 portfolios based on Price_Prediction
    forecast_data['Portfolio'] = forecast_data.groupby('Date')['Price_Prediction'].transform(
        lambda x: pd.qcut(x, n_bins, labels=False, duplicates='drop')
    )

    # Initialize a dictionary to store portfolios
    portfolios = {}

    # Loop through each portfolio (0 to N-1)
    for portfolio in range(n_bins):
        # Create a portfolio
        portfolio_data = forecast_data[forecast_data['Portfolio'] == portfolio][["Date", "Stock", "Price_Prediction"]]

        # Group by Date and create a list of stocks for each date
        grouped_portfolio = portfolio_data.groupby('Date').agg({
            'Stock': list,
            'Date': 'first'  # To keep the Date in the result
        }).reset_index(drop=True)

        if(portfolio == n_bins-1):
            last_picks = grouped_portfolio.copy()

        # Iterate through the DataFrame in steps of forecasting_horizon rows to fill the stock list
        for i in range(0, len(grouped_portfolio), forecasting_horizon):
            stock_list = grouped_portfolio.loc[i, 'Stock']
            for j in range(i, min(i + forecasting_horizon, len(grouped_portfolio))):
                grouped_portfolio.at[j, 'Stock'] = stock_list

        # Add returns for each portfolio
        grouped_portfolio['Returns'] = grouped_portfolio.apply(
            lambda row: stock_returns.loc[row['Date'], row['Stock']].tolist(), axis=1
        )

        # Calculate mean returns and cumulative product
        grouped_portfolio["Mean"] = grouped_portfolio["Returns"].apply(np.mean)
        grouped_portfolio["Cumprod"] = (grouped_portfolio["Mean"] + 1).cumprod()

        # Store the portfolio in the dictionary
        portfolios[portfolio] = grouped_portfolio

    # Labels for the portfolios
    labels = [f"Portfolio {i+1}" for i in range(n_bins)]

    # Create a color map
    colors = plt.cm.rainbow(np.linspace(0, 1, n_bins))[::-1]
    color_map = dict(zip(labels, colors))

    # Plotting
    plt.figure(figsize=(14, 7))

    # Plot each portfolio
    for portfolio, label in enumerate(labels):
        if portfolio == n_bins - 1:  # This is the best portfolio
            plt.plot(portfolios[portfolio]["Cumprod"], label=f"{label} (Best)", color=color_map[label], linewidth=3)
        else:
            plt.plot(portfolios[portfolio]["Cumprod"], label=label, color=color_map[label], alpha=0.7)

    # Calculate and plot the average of all portfolios
    avg_portfolio = pd.DataFrame(index=portfolios[0].index)
    for portfolio in portfolios.values():
        avg_portfolio = avg_portfolio.join(portfolio["Cumprod"], rsuffix=f'_{portfolio}')
    avg_portfolio['Average'] = avg_portfolio.mean(axis=1)
    plt.plot(avg_portfolio['Average'], label='THE MARKET', color='black', linewidth=4)

    # Add vertical lines for rebalancing points
    rebalance_dates = portfolios[0].index[::forecasting_horizon]
    for date in rebalance_dates:
        plt.axvline(x=date, color='gray', linestyle='--', alpha=0.5)

    # Calculate the step size for less frequent labels (e.g., every 30th label)
    step = len(portfolios[0]) // 30

    # Create less frequent tick locations and labels
    tick_locations = np.arange(0, len(portfolios[0]), step)
    tick_labels = portfolios[0]['Date'].iloc[tick_locations].dt.strftime('%Y-%m-%d')

    plt.xticks(tick_locations, tick_labels, rotation=45, ha='right')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small', ncol=2)

    plt.title('20 Portfolio Returns with Rebalancing Points and Average', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)

    plt.tight_layout()
    plt.show()

    best_portfolio_long = portfolios[n_bins-1]
    date = last_picks.iloc[-1]['Date']
    formatted_date = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date
    print(f"Best Picks to buy on date {formatted_date}, valid up to one month:")
    print(last_picks.iloc[-1]["Stock"])