import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet

# Function to convert 'K' and 'M' to numeric values
def convert_to_numeric(val):
    if isinstance(val, str):
        if 'K' in val:
            return float(val.replace('K', '')) * 1e3
        elif 'M' in val:
            return float(val.replace('M', '')) * 1e6
    return float(val)


def preprocess_dataframe(df, date_col='date', change_col='change %'):
    # Check if the date column exists
    if date_col in df.columns:
        # Convert date column to datetime and set as index
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.set_index(date_col)
    else:
        print(f"Warning: Column '{date_col}' not found in DataFrame.")
    
    # Remove '%' and convert 'change %' column to numeric
    if change_col in df.columns:
        df[change_col] = df[change_col].str.replace('%', '', regex=True).astype(float)
    
    return df


def calculate_metrics(test_values, predicted_values):
    """
    Function to calculate and display evaluation metrics for a model.
    
    Parameters:
    test_values (pd.Series or np.array): Actual values (ground truth)
    predicted_values (pd.Series or np.array): Predicted values by the model
    
    Returns:
    dict: A dictionary containing all the calculated metrics
    """
    # Calculate the metrics
    mae = mean_absolute_error(test_values, predicted_values)
    mse = mean_squared_error(test_values, predicted_values)
    rmse = mean_squared_error(test_values, predicted_values, squared=False)  # RMSE is the square root of MSE
    # mape = (abs((test_values - predicted_values) / test_values).mean()) * 100
    r2 = r2_score(test_values, predicted_values)
    
    # Display the results
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    # print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
    print(f"R-squared (R²): {r2}")
    
    # Return metrics as a dictionary for further use if needed
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        # "MAPE": mape,
        "R²": r2
    }

def expanding_window_cv_prophet(df, regressors, initial_train_size = 0.5, test_size = 90):
    """
    Perform expanding window cross-validation on a Prophet model with specified regressors.

    Parameters:
    - df (DataFrame): The time series data with 'ds' as date, 'y' as target, and additional regressors.
    - regressors (list): List of column names to be added as regressors.
    - initial_train_size (int): Initial size of the training set.
    - test_size (int): Number of observations in each test set.

    Returns:
    - avg_rmse (float): Average RMSE across all folds.
    - avg_mae (float): Average MAE across all folds.
    - rmse_scores (list): RMSE scores for each fold.
    - mae_scores (list): MAE scores for each fold.
    """
    rmse_scores = []
    mae_scores = []

    # Expanding window cross-validation loop
    for i in range(initial_train_size, len(df) - test_size, test_size):
        # Define training and testing sets with expanding window
        train_data = df.iloc[:i]
        test_data = df.iloc[i:i + test_size]
        
        # Initialize and configure the Prophet model with regressors
        model_fbp_lag_ma_cv = Prophet()
        for reg in regressors:
            model_fbp_lag_ma_cv.add_regressor(reg)
        
        # Fit the model on the expanding training set
        model_fbp_lag_ma_cv.fit(train_data)
        
        # Prepare future dataframe for the test period with necessary regressors
        future = test_data[['ds'] + regressors]
        
        # Predict on the test set
        forecast = model_fbp_lag_ma_cv.predict(future)
        predictions = forecast['yhat'].values
        
        # Compute the evaluation metrics (RMSE and MAE)
        rmse = np.sqrt(mean_squared_error(test_data['y'], predictions))
        mae = mean_absolute_error(test_data['y'], predictions)
        
        # Append the results for each fold
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        print(f"Fold ending at {test_data['ds'].iloc[-1]}: RMSE = {rmse}, MAE = {mae}")

    # Calculate average metrics across all folds
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    
    print("\nAverage RMSE:", avg_rmse)
    print("Average MAE:", avg_mae)
    
    return avg_rmse, avg_mae, rmse_scores, mae_scores


def plot_actual_vs_predicted(df_actual, df_predicted, actual_col='y', predicted_col='yhat', date_col='ds'):
    """
    Parameters:
    - df_actual (pd.DataFrame): DataFrame containing the actual values.
    - df_predicted (pd.DataFrame): DataFrame containing the predicted values.
    - actual_col (str): Column name for the actual values. Default is 'y'.
    - predicted_col (str): Column name for the predicted values. Default is 'yhat'.
    - date_col (str): Column name for the date values. Default is 'ds'.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    plt.plot(df_actual[date_col], df_actual[actual_col], label='Actual Price', color='blue')
    
    # Plot predicted values
    plt.plot(df_predicted[date_col], df_predicted[predicted_col], label='Predicted Price', color='orange')
    
    # Labeling
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Price (Prophet Model)')
    plt.legend()
    
    # Show plot
    plt.show()

