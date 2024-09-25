from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import joblib
import optuna
import warnings
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

N_TRIALS = 50  # Number of trials for Optuna optimization

def plot_scatter(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)

    #Mask removing the top 5% and bottom 5% of the outliers
    """mask = (y_true > np.percentile(y_true, 20)) & (y_true < np.percentile(y_true, 80))
    y_true = y_true[mask]
    y_pred = y_pred[mask]"""

    # Calculate the best fit line
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    line = slope * np.array(y_true) + intercept

    # Plot the best fit line
    plt.plot(y_true, line, 'r--', lw=2, label=f'Best fit line (slope: {slope:.4f})')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Scatter Plot of Actual vs Predicted Values')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Slope of the best fit line: {slope:.4f}")
    print(f"Intercept of the best fit line: {intercept:.4f}")

    # Calculate and print the angle of inclination
    angle = np.arctan(slope) * (180 / np.pi)
    print(f"Angle of inclination: {angle:.2f} degrees")

def prepare_data(training, validation, test):
    target_column = 'Target'  # Assuming 'Price' is the target for regression

    feature_columns = [col for col in training.columns if
                       col not in ['Date', 'Stock', target_column, 'Target', "Price"]]

    X_train = training[feature_columns]
    y_train = training[target_column]
    X_val = validation[feature_columns]
    y_val = validation[target_column]
    X_test = test[feature_columns]
    y_test = test[target_column]

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    #Save Date and stock
    test_date = test[['Date', 'Stock','Price']]

    return X_train_val, y_train_val, X_test, y_test, feature_columns, test_date

def objective(trial, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'num_leaves': trial.suggest_int('num_leaves', 2, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 200),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'verbose': -1
    }

    model = LGBMRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    non_zero_indices = y_pred != 0
    mape = np.mean(np.abs((y_pred[non_zero_indices] - y_val[non_zero_indices]) / y_pred[non_zero_indices])) * 100
    print(f"MAPE: {mape:.2f}%")

    return mape

def early_stopping_callback(study, trial):
    if study.best_trial.number + 10 < trial.number:
        study.stop()

def train_model(X_train_val, y_train_val):
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, X_train_val, y_train_val), n_trials=N_TRIALS, callbacks=[early_stopping_callback])
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()

    trial = study.best_trial
    best_params = trial.params

    joblib.dump(best_params, 'best_params.joblib')
    best_model = LGBMRegressor(**best_params, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    eval_set = [(X_train, y_train), (X_val, y_val)]

    # Fit the model with evaluation set and early stopping
    best_model.fit(X_train, y_train,
                   eval_set=eval_set,
                   eval_metric='mape')  # Set verbose to 100 to see output every 100 iterations

    # Extract training and validation scores
    train_scores = best_model.evals_result_['training']['mape']
    val_scores = best_model.evals_result_['valid_1']['mape']

    # Plot training and validation errors
    plt.figure(figsize=(10, 6))
    plt.plot(train_scores, label='Training error')
    plt.plot(val_scores, label='Validation error')
    plt.axvline(x=best_model.best_iteration_, color='r', linestyle='--', label='Best iteration')
    plt.xlabel('Iterations')
    plt.ylabel('Error metric')
    plt.title('Best Model: Training and Validation Error')
    plt.legend()
    plt.show()

    # Fit the final model on all data
    best_model.fit(X_train_val, y_train_val)

    return best_model



def evaluate_model(model, X_test, y_test):
    test_predictions = model.predict(X_test)
    non_zero_indices = y_test != 0
    mape = np.mean(np.abs((y_test[non_zero_indices] - test_predictions[non_zero_indices]) / y_test[non_zero_indices])) * 100

    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return test_predictions, X_test

def print_feature_importance(model, feature_columns):
    # Get feature importances
    importances = model.feature_importances_

    # Normalize the importances
    normalized_importances = importances / np.sum(importances)

    # Create a DataFrame with features and their importances
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances,
        'normalized_importance': normalized_importances
    })

    # Sort by normalized importance in descending order
    feature_importance = feature_importance.sort_values('normalized_importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']}: {row['normalized_importance']:.4f} ({row['importance']:.4f})")

    # Optionally, you can create a bar plot of the top features
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'].head(10), feature_importance['normalized_importance'].head(10))
    plt.title('Top 10 Feature Importances (Normalized)')
    plt.xlabel('Features')
    plt.ylabel('Normalized Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

def main_forecasting(training, validation, test,MODEL_PATH, MODE, index, years, n_assets):


    X_train_val, y_train_val, X_test, y_test, feature_columns, test_date = prepare_data(training, validation, test)

    if MODE == 'train':
        model = train_model(X_train_val, y_train_val)
        model_filename = MODEL_PATH
        save_model(model, model_filename)
    elif MODE == 'load':
        # Assume the latest model file is to be loaded
        model = load_model(MODEL_PATH)

    test_predictions, X_test = evaluate_model(model, X_test, y_test)

    print_feature_importance(model, feature_columns)

    # Re-combine the test_date and predictions
    test_predictions = pd.DataFrame(test_predictions, columns=['Price_Prediction'])
    test_predictions = pd.concat([test_date, test_predictions], axis=1)

    stock_returns = test_predictions[["Date", "Stock", "Price"]]

    forecast_data = test_predictions#[test_predictions['Confidence'] > 0].copy()

    stock_returns = stock_returns.pivot(index='Date', columns='Stock', values='Price')
    forecast_data = forecast_data.pivot(index='Date', columns='Stock', values='Price_Prediction')

    stock_returns = stock_returns.pct_change().dropna()
    forecast_data = forecast_data.pct_change().dropna()
    plot_scatter(stock_returns.values.flatten(), forecast_data.values.flatten())

    return forecast_data,stock_returns

