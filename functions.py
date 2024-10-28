import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Example of how to use the function
# metrics = calculate_metrics(test_fbp['y'], predictions_fbp['yhat'])

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

# Example usage
# plot_actual_vs_predicted(df_n, predictions_fbp_lag)
