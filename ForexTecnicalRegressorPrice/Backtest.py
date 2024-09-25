import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def preprocess_forecast_data(forecast_data):
    """Preprocess the forecast data."""
    return forecast_data.reset_index().melt(id_vars='Date', var_name='Stock', value_name='Price_Prediction')


def create_quintile_portfolios(forecast_data):
    """Create 5 portfolios (quintiles) based on Price_Prediction."""
    forecast_data['Quintile'] = forecast_data.groupby('Date')['Price_Prediction'].transform(
        lambda x: pd.qcut(x, 5, labels=False)
    )
    return forecast_data


def create_portfolio(quintile_data, stock_returns, forecasting_horizon):
    """Create a portfolio for a given quintile."""
    portfolio = quintile_data[["Date", "Stock", "Price_Prediction"]]
    grouped_portfolio = portfolio.groupby('Date').agg({
        'Stock': list,
        'Date': 'first'
    }).reset_index(drop=True)

    for i in range(0, len(grouped_portfolio), forecasting_horizon):
        stock_list = grouped_portfolio.loc[i, 'Stock']
        for j in range(i, min(i + forecasting_horizon, len(grouped_portfolio))):
            grouped_portfolio.at[j, 'Stock'] = stock_list

    grouped_portfolio['Returns'] = grouped_portfolio.apply(
        lambda row: stock_returns.loc[row['Date'], row['Stock']].tolist(), axis=1
    )
    grouped_portfolio["Mean"] = grouped_portfolio["Returns"].apply(np.mean)
    grouped_portfolio["Cumprod"] = (grouped_portfolio["Mean"] + 1).cumprod()

    return grouped_portfolio


def calculate_portfolio_statistics(best_portfolio, worst_portfolio):
    """Calculate and print portfolio statistics."""
    best_cumprod = best_portfolio[['Date', 'Cumprod']]
    worst_cumprod = worst_portfolio[['Date', 'Cumprod']]

    best_cumprod.loc[:, 'Date'] = pd.to_datetime(best_cumprod['Date'])
    worst_cumprod.loc[:, 'Date'] = pd.to_datetime(worst_cumprod['Date'])

    best_cumprod.set_index('Date', inplace=True)
    worst_cumprod.set_index('Date', inplace=True)

    monthly_returns_best = best_cumprod.resample('M').last().pct_change().dropna()
    yearly_returns_best = best_cumprod.resample('Y').last().pct_change().dropna()

    monthly_returns_worst = worst_cumprod.resample('M').last().pct_change().dropna()
    yearly_returns_worst = worst_cumprod.resample('Y').last().pct_change().dropna()

    merged_monthly = pd.concat([monthly_returns_best, monthly_returns_worst], axis=1)
    merged_monthly.columns = ['Strong Buy', 'Strong Sell']

    count_strong_buy_greater = (merged_monthly['Strong Buy'] > merged_monthly['Strong Sell']).sum()
    total_count = len(merged_monthly)
    percentage_strong_buy_greater = (count_strong_buy_greater / total_count) * 100

    merged_yearly = pd.concat([yearly_returns_best, yearly_returns_worst], axis=1)
    merged_yearly.columns = ['Strong Buy', 'Strong Sell']

    count_strong_buy_greater_yearly = (merged_yearly['Strong Buy'] > merged_yearly['Strong Sell']).sum()
    total_count_yearly = len(merged_yearly)
    percentage_strong_buy_greater_yearly = (count_strong_buy_greater_yearly / total_count_yearly) * 100

    print(f"Percentage of times Strong Buy > Strong Sell (Monthly): {percentage_strong_buy_greater:.2f}%")
    print(f"Percentage of times Strong Buy > Strong Sell (Yearly): {percentage_strong_buy_greater_yearly:.2f}%")

    average_monthly_return_best = monthly_returns_best.mean().values[0]
    print(f"Average monthly return for the best portfolio: {average_monthly_return_best:.2f}")

    average_yearly_return_best = yearly_returns_best.mean().values[0]
    print(f"Average yearly return for the best portfolio: {average_yearly_return_best:.2f}")


def plot_quintile_portfolios(portfolios, forecasting_horizon):
    """Plot the quintile portfolios."""
    labels = ["Strong Buy", "Strong Sell"]
    labels.reverse()
    colors = ['darkgreen', 'red']
    colors.reverse()
    color_map = dict(zip(labels, colors))

    plt.figure(figsize=(14, 7))

    for quintile, label in enumerate(labels):
        plt.plot(portfolios[quintile]["Cumprod"], label=label, color=color_map[label])

    rebalance_dates = portfolios[0].index[::forecasting_horizon]
    for date in rebalance_dates:
        plt.axvline(x=date, color='gray', linestyle='--', alpha=0.5)

    step = len(portfolios[0]) // 30
    tick_locations = np.arange(0, len(portfolios[0]), step)
    tick_labels = portfolios[0]['Date'].iloc[tick_locations].dt.strftime('%Y-%m-%d')

    plt.xticks(tick_locations, tick_labels, rotation=45, ha='right')
    plt.legend()
    plt.title('Quintile Portfolio Returns with Rebalancing Points')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.tight_layout()
    plt.show()


def main_backtest(forecast_data, stock_returns,n_weeks, n_bins=2):
    """Main function to run the backtest."""
    forecasting_horizon = n_weeks*7

    forecast_data = preprocess_forecast_data(forecast_data)
    forecast_data = create_quintile_portfolios(forecast_data)

    portfolios = {}
    for quintile in range(n_bins):
        quintile_data = forecast_data[forecast_data['Quintile'] == quintile]
        portfolios[quintile] = create_portfolio(quintile_data, stock_returns, forecasting_horizon)

    calculate_portfolio_statistics(portfolios[n_bins-1], portfolios[0])
    plot_quintile_portfolios(portfolios, forecasting_horizon)

