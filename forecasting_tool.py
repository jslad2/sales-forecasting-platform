import pandas as pd
import streamlit as st
from prophet import Prophet
from pmdarima import auto_arima
from xgboost import XGBRegressor
from flaml import AutoML
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import plotly.express as px
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import optuna
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf
import time
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import catboost
from tqdm import tqdm
import concurrent.futures
from scipy.stats import pearsonr

# Enable Wide Mode (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(layout="wide", page_title="Time Series Forecasting", page_icon="📈")

# Add dark mode toggle
if "theme" not in st.session_state:
    st.session_state.theme = "light"

theme = st.radio("🌙 Theme Mode:", ["Light", "Dark"], index=0 if st.session_state.theme == "light" else 1)
st.session_state.theme = theme

if st.session_state.theme == "dark":
    st.markdown(
        """
        <style>
            body { background-color: #1E1E1E; color: white; }
            .stButton button { background-color: #56BBAF !important; }
            .stDataFrame { background-color: #2E2E2E; color: white; }
            .stTextInput input { background-color: #2E2E2E; color: white; }
            .stSelectbox select { background-color: #2E2E2E; color: white; }
            .stRadio div { color: white; }
            .stMarkdown { color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
            body { background-color: white; color: black; }
            .stButton button { background-color: #56BBAF !important; }
            .stDataFrame { background-color: white; color: black; }
            .stTextInput input { background-color: white; color: black; }
            .stSelectbox select { background-color: white; color: black; }
            .stRadio div { color: black; }
            .stMarkdown { color: black; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Load environment variables from .env
# load_dotenv()

# Initialize Supabase Client
# supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def is_feature_available(subscription_level, feature):
    subscription_levels = {
        "Free": {
            "Basic Forecasting": True,
            "Best Model Selection": False,
            "AutoML Hyperparameter Tuning": False,
            "Custom Forecast Intervals": False,
            "Scenario Planning & Demand Shocks": False,
            "Business Impact Insights": False,
            "Anomaly Detection": False,
            "Alerts & Monitoring": False,
            "Google Sheets & Excel Exports": False,
            "Multi-User Access & API Integration": False
        },
        "basic": {
            "Basic Forecasting": True,
            "Best Model Selection": True,
            "AutoML Hyperparameter Tuning": False,
            "Custom Forecast Intervals": True,
            "Scenario Planning & Demand Shocks": False,
            "Business Impact Insights": True,
            "Anomaly Detection": False,
            "Alerts & Monitoring": False,
            "Google Sheets & Excel Exports": True,
            "Multi-User Access & API Integration": False
        },
        "premium": {
            "Basic Forecasting": True,
            "Best Model Selection": True,
            "AutoML Hyperparameter Tuning": True,
            "Custom Forecast Intervals": True,
            "Scenario Planning & Demand Shocks": True,
            "Business Impact Insights": True,
            "Anomaly Detection": True,
            "Alerts & Monitoring": True,
            "Google Sheets & Excel Exports": True,
            "Multi-User Access & API Integration": True
        }
    }
    return subscription_levels.get(subscription_level, {}).get(feature, False)

def check_stationarity(series):
    adf_result = adfuller(series, autolag="AIC")
    adf_p_value = adf_result[1]
    kpss_result = kpss(series, regression="c", nlags="auto")
    kpss_p_value = kpss_result[1]
    if adf_p_value < 0.05 and kpss_p_value > 0.05:
        return "Stationary"
    elif adf_p_value >= 0.05 and kpss_p_value <= 0.05:
        return "Non-Stationary"
    else:
        return "Inconclusive"

def preprocess_data(data, date_column, sales_column, category_columns=None):
    """
    Preprocess the uploaded data, check stationarity, and apply transformations if needed.
    Returns a dataframe with columns "ds" and "y" (and optionally categories) for Prophet compatibility.
    Also returns the last historical value before forecasting if differencing is applied.
    """
    try:
        # 1. Convert date column to datetime
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data.dropna(subset=[date_column, sales_column], inplace=True)

        # 2. Rename columns for Prophet compatibility
        data = data.rename(columns={date_column: "ds", sales_column: "y"})

        # 3. Create a 'monthly' period column
        data["year_month"] = data["ds"].dt.to_period("M")

        # 4. Handle category columns
        if category_columns:
            # Ensure category_columns is a list (even if it's a single column)
            if not isinstance(category_columns, list):
                category_columns = [category_columns]

            # Group by monthly period + category columns
            grouping_cols = ["year_month"] + category_columns
        else:
            # Group by monthly period only
            grouping_cols = ["year_month"]

        # 5. Aggregate (sum) sales at the monthly level
        data = data.groupby(grouping_cols, as_index=False)["y"].sum()

        # 6. Convert 'year_month' back to a proper datetime (start of each month)
        data["ds"] = data["year_month"].dt.to_timestamp()
        data.drop(columns=["year_month"], inplace=True)

        # 7. Check for duplicate dates (and categories) after aggregation
        subset_cols = ["ds"] + (category_columns if category_columns else [])
        if data.duplicated(subset=subset_cols).any():
            st.warning("⚠️ Duplicate date/category combinations found. Removing duplicates...")
            data = data.drop_duplicates(subset=subset_cols, keep="last")

        # 8. Store original values (for plotting/inspection later)
        y_original = data[["ds", "y"]].copy().rename(columns={"y": "y_original"})

        # 9. Check stationarity of the aggregated series
        stationarity_result = check_stationarity(data["y"])
        st.markdown(
            f"""
            <div style="text-align: center;">
                <h2 style="color: #2B3A42;">📊 Stationarity Test</h2>
                <p style="font-size: 1.2rem;">Conclusion: The series is <strong>{stationarity_result}</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 10. If non-stationary, apply differencing
        if stationarity_result == "Non-Stationary":
            st.warning("Applying differencing to stabilize the series.")

            data["y_diff"] = data["y"].diff()
            last_historical_value = data["y"].iloc[-1]

            # Visualization: Original vs. Differenced
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["ds"],
                y=data["y"],
                mode="lines",
                name="Original Series",
                line=dict(color="blue", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=data["ds"].iloc[1:],
                y=data["y_diff"].dropna(),
                mode="lines",
                name="Differenced Series",
                line=dict(color="orange", width=2, dash="dot")
            ))
            fig.update_layout(
                title="Original vs Differenced Series (Monthly)",
                xaxis_title="Date",
                yaxis_title="Sales",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Remove NaNs from differencing
            differenced_data = data.dropna(subset=["y_diff"]).drop(columns=["y"])
            differenced_data = differenced_data.rename(columns={"y_diff": "y"})
            differenced_data = differenced_data.reset_index(drop=True)
            differenced_data.columns = differenced_data.columns.astype(str)

            return differenced_data[["ds", "y"]], last_historical_value, y_original

        else:
            # If already stationary, just visualize the original monthly series
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["ds"],
                y=data["y"],
                mode="lines",
                name="Original Series (Monthly)",
                line=dict(color="blue", width=2)
            ))
            fig.update_layout(
                title="📊 Original Series (Monthly)",
                xaxis_title="Date",
                yaxis_title="Sales",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Return the monthly-aggregated data without differencing
            data = data.reset_index(drop=True)
            return data[["ds", "y"]], None, y_original

    except Exception as e:
        st.error(f"An error occurred during preprocessing: {e}")
        st.error(f"Debug Info: Columns in data - {data.columns}, Data Shape - {data.shape}")
        return None, None, None

def inverse_difference(forecast_data, first_value):
    """
    Reverse differencing to restore original scale.
    
    :param forecast_data: DataFrame or Series containing the forecasted values.
    :param first_value: The last historical value before differencing.
    :return: DataFrame or Series with restored values.
    """
    if first_value is not None:
        # Ensure first_value is numeric
        if not isinstance(first_value, (int, float)):
            raise ValueError("first_value must be a numeric value (int or float).")

        # Convert Series to DataFrame if necessary
        if isinstance(forecast_data, pd.Series):
            forecast_data = forecast_data.to_frame(name="yhat")

        # Restore original values using cumulative sums
        if "yhat" in forecast_data.columns:
            forecast_data["yhat"] = first_value + forecast_data["yhat"].cumsum()
        
        # Restore upper and lower bounds if they exist
        if "yhat_upper" in forecast_data.columns:
            forecast_data["yhat_upper"] = first_value + forecast_data["yhat_upper"].cumsum()
        
        if "yhat_lower" in forecast_data.columns:
            forecast_data["yhat_lower"] = first_value + forecast_data["yhat_lower"].cumsum()
    
    return forecast_data

    # if last_historical_value is not None:
    #     prophet_forecast["yhat"] = last_historical_value + prophet_forecast["yhat"].cumsum() - prophet_forecast["yhat"].iloc[0]
    #     prophet_forecast["yhat_upper"] = last_historical_value + prophet_forecast["yhat_upper"].cumsum() - prophet_forecast["yhat_upper"].iloc[0]
    #     prophet_forecast["yhat_lower"] = last_historical_value + prophet_forecast["yhat_lower"].cumsum() - prophet_forecast["yhat_lower"].iloc[0]

def detect_and_add_seasonalities(model, data):
    data_frequency = pd.infer_freq(data["ds"])
    if data_frequency == "D":
        model.add_seasonality(name="daily", period=1, fourier_order=3)
    elif data_frequency == "W":
        model.add_seasonality(name="weekly", period=7, fourier_order=3)
    elif data_frequency == "M":
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    elif data_frequency == "Q":
        model.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
    elif data_frequency == "Y":
        model.add_seasonality(name="yearly", period=365.25, fourier_order=10)
    return model

def evaluate_prophet_params(params, train, initial, horizon, period):
    try:
        model = Prophet(
            seasonality_mode=params["seasonality_mode"],
            changepoint_prior_scale=params["changepoint_prior_scale"],
            yearly_seasonality=(len(train) >= 365),
            weekly_seasonality=(len(train) >= 30),
            daily_seasonality=False
        )
        model.fit(train)
        cv_results = cross_validation(model, initial=initial, horizon=horizon, period=period)
        metrics = performance_metrics(cv_results)
        rmse = metrics["rmse"].mean()
        return (params, rmse)
    except Exception as e:
        st.write(f"Failed with params {params}: {e}")
        return (params, float("inf"))

def find_best_prophet_params(train):
    dataset_length = len(train)
    if dataset_length < 100:
        param_grid = {
            "changepoint_prior_scale": [0.01, 0.1],
            "seasonality_mode": ["additive"]
        }
    else:
        param_grid = {
            "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.2, 0.3],
            "seasonality_mode": ["additive", "multiplicative"]
        }
    if dataset_length < 100:
        horizon_days = min(7, max(3, dataset_length // 5))
        initial_days = max(30, dataset_length // 2)
    else:
        horizon_days = min(30, max(7, dataset_length // 10))
        initial_days = max(90, dataset_length // 3)
    horizon = f"{horizon_days} days"
    initial = f"{initial_days} days"
    period = f"{horizon_days // 2} days"
    
    best_params = None
    best_rmse = float("inf")
    
    # Use a ThreadPoolExecutor to evaluate parameter combinations concurrently
    grid = list(ParameterGrid(param_grid))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(evaluate_prophet_params, params, train, initial, horizon, period): params for params in grid}
        for future in concurrent.futures.as_completed(futures):
            params, rmse = future.result()
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
    return best_params, best_rmse

def apply_scenarios(data, demand_shock, seasonality_adjustment, external_shock, category_columns=None, category_adjustments=None, category_date_ranges=None):
    """
    Apply scenario adjustments to the training data.
    Supports global adjustments (demand shock, seasonality, external shock) and category-specific adjustments.
    """
    try:
        # Validate input data
        if data is None or data.empty:
            st.error("❌ No data provided. Please check your input.")
            return data

        # Ensure required columns are present
        if "ds" not in data.columns or "y" not in data.columns:
            st.error("❌ Required columns 'ds' (date) and 'y' (sales) are missing.")
            return data

        # Apply global demand shock
        if demand_shock != 0:
            data["y"] = data["y"] * (1 + demand_shock / 100)
            st.write(f"✅ Applied global demand shock: {demand_shock}%")

        # Apply global seasonality adjustment
        if seasonality_adjustment != 0:
            data["month"] = data["ds"].dt.month
            seasonality_multiplier = 1 + seasonality_adjustment / 100
            data["y"] = data["y"] * (1 + (data["month"] - 1) * (seasonality_multiplier - 1) / 12)
            st.write(f"✅ Applied global seasonality adjustment: {seasonality_adjustment}%")

        # Apply global external shock
        if external_shock:
            data["y"] = data["y"] * 0.8  # Simulate a 20% reduction in sales
            st.write("✅ Applied global external shock: 20% reduction in sales")

        # Apply category-specific adjustments if categories are selected
        if category_columns and category_adjustments and category_date_ranges:
            st.write("🔍 Applying category-specific adjustments...")
            
            for category_column, category_adjustment in zip(category_columns, category_adjustments):
                if category_column in data.columns:
                    # Get the date range for the current category
                    start_date, end_date = category_date_ranges[category_column]

                    # Convert start_date and end_date to datetime
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)

                    # Filter data for the selected time period
                    scenario_data = data[
                        (data["ds"] >= start_date) & 
                        (data["ds"] <= end_date)
                    ]
                    
                    # Apply adjustments to all categories in the column
                    for category in scenario_data[category_column].unique():
                        # Increase or decrease sales for the selected category
                        scenario_data.loc[scenario_data[category_column] == category, "y"] *= (1 + category_adjustment / 100)
                        st.write(f"✅ Applied {category_adjustment}% adjustment to {category_column}: {category} (from {start_date.date()} to {end_date.date()})")

                    # Update the main data with the adjusted values
                    data.update(scenario_data)
                else:
                    st.warning(f"⚠️ Column '{category_column}' not found in the dataset. Skipping adjustments for this category.")
        elif category_columns:
            st.warning("⚠️ No time period selected for category-specific adjustments. Skipping.")

        # Return the modified data
        return data

    except Exception as e:
        st.error(f"❌ An error occurred while applying scenarios: {e}")
        st.error(f"Debug Info: Columns in data - {data.columns}, Data Shape - {data.shape}")
        return data  # Return the original data in case of an error

def train_prophet_model(train, test, forecast_period, best_params, last_historical_value, y_original):
    result = {}
    try:
        model = Prophet(
            seasonality_mode=best_params["seasonality_mode"],
            changepoint_prior_scale=best_params["changepoint_prior_scale"]
        )
        try:
            model = detect_and_add_seasonalities(model, train)
        except Exception as e:
            st.warning(f"Seasonality detection failed: {e}. Proceeding without additional seasonalities.")
        model.fit(train)
        future = model.make_future_dataframe(periods=forecast_period, freq="M", include_history=False)
        forecast = model.predict(future)
        forecast = forecast[forecast["ds"] > train["ds"].max()]
        matching_length = min(len(test["y"]), len(forecast))
        rmse = mean_squared_error(test["y"].iloc[:matching_length], forecast["yhat"].iloc[:matching_length]) ** 0.5
        mape = mean_absolute_percentage_error(test["y"].iloc[:matching_length], forecast["yhat"].iloc[:matching_length])
        if isinstance(last_historical_value, (int, float)):
            forecast = inverse_difference(forecast, last_historical_value)
        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast}
    except Exception as e:
        st.warning(f"Prophet Model failed: {e}")
    return ("Prophet", result)

def train_arima_model(train, test, forecast_period, last_historical_value, y_original):
    result = {}
    try:
        try:
            decomposition = seasonal_decompose(train["y"], model="additive", period=12)
            seasonality_present = np.any(np.abs(decomposition.seasonal) > 0.01)
            acf_values = acf(train["y"], nlags=12, fft=False)
            seasonality_confirmed = any(np.abs(acf_values[1:]) > 0.2)
            seasonal = seasonality_present and seasonality_confirmed
        except Exception as e:
            st.warning(f"Error in seasonality analysis: {e}")
            seasonal = False
        model = auto_arima(
            train["y"],
            seasonal=seasonal,
            m=12 if seasonal else 1,
            d=None,
            D=1 if seasonal else 0,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            trace=True,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True
        )
        arima_forecast = model.predict(n_periods=forecast_period)
        if len(arima_forecast) < forecast_period:
            forecast_period = len(arima_forecast)
        forecast_dates = pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1), periods=forecast_period, freq="M")
        forecast_df = pd.DataFrame({"ds": forecast_dates, "yhat": arima_forecast[:forecast_period]})
        if isinstance(last_historical_value, (int, float)):
            forecast_df["yhat"] = inverse_difference(forecast_df["yhat"], last_historical_value)
        matching_length = min(len(test["y"]), len(forecast_df))
        rmse = mean_squared_error(test["y"].iloc[:matching_length], forecast_df["yhat"].iloc[:matching_length]) ** 0.5
        mape = mean_absolute_percentage_error(test["y"].iloc[:matching_length], forecast_df["yhat"].iloc[:matching_length])
        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast_df}
    except Exception as e:
        st.warning(f"ARIMA Model failed: {e}")
    return ("ARIMA", result)

def train_xgb_model(train, test, forecast_period, last_historical_value, y_original):
    result = {}
    try:
        max_lag = min(12, len(train) - 1)
        rolling_windows = [3, 6] if len(train) > 6 else [3]
        data_xgb = train.copy()
        for lag in range(1, max_lag + 1):
            data_xgb[f"lag_{lag}"] = data_xgb["y"].shift(lag)
        for window in rolling_windows:
            data_xgb[f"rolling_mean_{window}"] = data_xgb["y"].rolling(window=window).mean()
            data_xgb[f"rolling_std_{window}"] = data_xgb["y"].rolling(window=window).std()
        data_xgb["month"] = data_xgb["ds"].dt.month
        data_xgb["quarter"] = data_xgb["ds"].dt.quarter
        data_xgb["year"] = data_xgb["ds"].dt.year
        data_xgb["sin_month"] = np.sin(2 * np.pi * data_xgb["month"] / 12)
        data_xgb["cos_month"] = np.cos(2 * np.pi * data_xgb["month"] / 12)
        data_xgb.dropna(inplace=True)
        feature_cols = [col for col in data_xgb.columns if col not in ["y", "ds"]]
        X_train = data_xgb[feature_cols]
        y_train = data_xgb["y"]
        model = XGBRegressor(
            n_estimators=50,
            max_depth=min(5, max(2, len(train) // 10)),
            learning_rate=0.1 if len(train) > 50 else 0.2,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        future_features = []
        for i in range(forecast_period):
            future_row = {}
            for lag in range(1, max_lag + 1):
                if lag == 1:
                    future_row[f"lag_{lag}"] = data_xgb["y"].iloc[-1]
                else:
                    future_row[f"lag_{lag}"] = data_xgb[f"lag_{lag - 1}"].iloc[-1]
            for window in rolling_windows:
                future_row[f"rolling_mean_{window}"] = data_xgb[f"rolling_mean_{window}"].iloc[-1]
                future_row[f"rolling_std_{window}"] = data_xgb[f"rolling_std_{window}"].iloc[-1]
            future_row["month"] = (data_xgb["ds"].iloc[-1] + pd.DateOffset(months=i + 1)).month
            future_row["quarter"] = (data_xgb["ds"].iloc[-1] + pd.DateOffset(months=i + 1)).quarter
            future_row["year"] = (data_xgb["ds"].iloc[-1] + pd.DateOffset(months=i + 1)).year
            future_row["sin_month"] = np.sin(2 * np.pi * future_row["month"] / 12)
            future_row["cos_month"] = np.cos(2 * np.pi * future_row["month"] / 12)
            future_features.append(future_row)
        future_df = pd.DataFrame(future_features)
        future_df.fillna(method="ffill", inplace=True)
        xgb_forecast = model.predict(future_df)
        matching_length = min(len(test["y"]), len(xgb_forecast))
        rmse = mean_squared_error(test["y"].iloc[:matching_length], xgb_forecast[:matching_length]) ** 0.5
        mape = mean_absolute_percentage_error(test["y"].iloc[:matching_length], xgb_forecast[:matching_length])
        forecast_df = pd.DataFrame({
            "ds": pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1), periods=forecast_period, freq="M"),
            "yhat": xgb_forecast
        })
        if isinstance(last_historical_value, (int, float)):
            forecast_df["yhat"] = inverse_difference(forecast_df["yhat"], last_historical_value)
        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast_df}
    except Exception as e:
        st.warning(f"XGBoost Model failed: {e}")
    return ("XGBoost", result)

def train_automl_model(train, test, forecast_period, last_historical_value, y_original, time_budget=None):
    """
    Train an AutoML model using FLAML for time series forecasting.

    Args:
        train (pd.DataFrame): Training data with columns "ds" (date) and "y" (target).
        test (pd.DataFrame): Test data for evaluation.
        forecast_period (int): Number of periods to forecast.
        last_historical_value (float): Last observed value before differencing (if applied).
        y_original (pd.DataFrame): Original target values for comparison.
        time_budget (int or None): Time budget in seconds for AutoML training. If None, it is dynamically calculated.

    Returns:
        tuple: Model name ("AutoML") and a dictionary containing RMSE, MAPE, and forecast DataFrame.
    """
    result = {}
    try:
        # Feature Engineering
        data_automl = train.copy()

        # Determine maximum lag based on dataset size
        max_lag = min(24, len(train) - 1)  # Cap at 24 lags
        if len(train) <= 6:
            max_lag = min(3, len(train) - 1)
        elif len(train) <= 12:
            max_lag = min(6, len(train) - 1)
        elif len(train) <= 24:
            max_lag = min(12, len(train) - 1)

        # Add lag features
        for lag in range(1, max_lag + 1):
            data_automl[f"lag_{lag}"] = data_automl["y"].shift(lag)

        # Add rolling statistics
        for window in [3, 6, 12]:
            data_automl[f"rolling_mean_{window}"] = data_automl["y"].rolling(window=window, min_periods=1).mean()
            data_automl[f"rolling_std_{window}"] = data_automl["y"].rolling(window=window, min_periods=1).std()

        # Add year-over-year growth (if sufficient data)
        if len(train) > 12:
            data_automl["yoy_growth"] = (data_automl["y"] / data_automl["y"].shift(12)) - 1
        else:
            data_automl["yoy_growth"] = 0

        # Add differenced values
        data_automl["y_diff"] = data_automl["y"].diff().fillna(0)

        # Add rolling mean growth
        data_automl["rolling_mean_growth"] = data_automl["y"].rolling(window=3).mean().diff().fillna(0)

        # Add trigonometric features for seasonality
        data_automl["sin_month"] = np.sin(2 * np.pi * data_automl["ds"].dt.month / 12)
        data_automl["cos_month"] = np.cos(2 * np.pi * data_automl["ds"].dt.month / 12)

        # Log-transform if the target variable has a large range
        if data_automl["y"].max() / data_automl["y"].min() > 5:
            data_automl["y_log"] = np.log1p(data_automl["y"])
            apply_log = True
        else:
            data_automl["y_log"] = data_automl["y"]
            apply_log = False

        # Drop rows with missing values
        data_automl.dropna(inplace=True)

        # Prepare features and target
        feature_cols = [col for col in data_automl.columns if col not in ["y", "ds", "y_log"]]
        y_train = data_automl["y_log"] if apply_log else data_automl["y"]
        X_train = data_automl[feature_cols]

        # Dynamic time budget calculation (if not provided)
        if time_budget is None:
            # Base time budget on dataset size and number of features
            time_budget = min(600, max(60, len(train) * 0.1 + len(feature_cols) * 2))  # 60s to 600s
            st.info(f"Dynamic time budget set to {time_budget} seconds based on dataset size and complexity.")

        # Train AutoML model
        automl_model = AutoML()
        automl_model.fit(
            X_train=X_train,
            y_train=y_train,
            task="regression",
            time_budget=time_budget,
            eval_method="cv",
            estimator_list=["xgboost", "lgbm", "rf", "catboost"],
            metric="r2",
            early_stop=True,  # Enable early stopping
            verbose=1  # Show progress
        )

        # Generate future features for forecasting
        future_features = []
        last_row = data_automl.iloc[-1].copy()
        for i in range(forecast_period):
            future_row = {}
            for lag in range(1, max_lag + 1):
                if lag == 1:
                    future_row[f"lag_{lag}"] = last_row["y_log"] if apply_log else last_row["y"]
                else:
                    future_row[f"lag_{lag}"] = last_row[f"lag_{lag - 1}"]
            for window in [3, 6, 12]:
                if apply_log:
                    future_row[f"rolling_mean_{window}"] = last_row[f"rolling_mean_{window}"] + (last_row["y_log"] - last_row[f"lag_{window}"]) / window
                else:
                    future_row[f"rolling_mean_{window}"] = last_row[f"rolling_mean_{window}"] + (last_row["y"] - last_row[f"lag_{window}"]) / window
                future_row[f"rolling_std_{window}"] = last_row[f"rolling_std_{window}"]
            future_row["yoy_growth"] = last_row["yoy_growth"]
            future_row["y_diff"] = last_row["y_diff"]
            future_row["rolling_mean_growth"] = last_row["rolling_mean_growth"]
            future_row["sin_month"] = np.sin(2 * np.pi * (last_row["ds"].month + i) / 12)
            future_row["cos_month"] = np.cos(2 * np.pi * (last_row["ds"].month + i) / 12)
            future_features.append(future_row)
            last_row = last_row.copy()
            for key, value in future_row.items():
                last_row[key] = value

        # Create future DataFrame
        future_df = pd.DataFrame(future_features)
        for col in X_train.columns:
            if col not in future_df.columns:
                future_df[col] = 0
        future_df = future_df[X_train.columns]

        # Generate forecasts
        automl_forecast = automl_model.predict(future_df)
        if apply_log:
            automl_forecast = np.expm1(automl_forecast)

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            "ds": pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1), periods=forecast_period, freq="M"),
            "yhat": automl_forecast,
            "yhat_lower": automl_forecast * 0.9,
            "yhat_upper": automl_forecast * 1.1
        })

        # Inverse differencing if applicable
        if isinstance(last_historical_value, (int, float)):
            forecast_df["yhat"] = inverse_difference(forecast_df["yhat"], last_historical_value)
            forecast_df["yhat_lower"] = inverse_difference(forecast_df["yhat_lower"], last_historical_value)
            forecast_df["yhat_upper"] = inverse_difference(forecast_df["yhat_upper"], last_historical_value)

        # Evaluate model performance
        matching_length = min(len(test["y"]), len(forecast_df))
        rmse = np.sqrt(mean_squared_error(test["y"].values, forecast_df["yhat"][:len(test["y"])]))
        mape = mean_absolute_percentage_error(test["y"].values, forecast_df["yhat"][:len(test["y"])])
        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast_df}

    except Exception as e:
        st.error(f"AutoML Model failed: {e}")
    return ("AutoML", result)

def shape_score(actual, forecast):
    m = min(len(actual), len(forecast))
    if m == 0:
        return 0
    actual_series = actual.iloc[:m]
    forecast_series = forecast.iloc[:m]
    corr, _ = pearsonr(actual_series, forecast_series)
    return corr

def combined_score(rmse, shape, max_rmse, alpha, beta):
    norm_rmse = rmse / max_rmse if max_rmse != 0 else 0
    return alpha * norm_rmse + beta * (1 - shape)

def main():
    user_id = "user123"
    subscription_level = "premium"
    if subscription_level != "premium":
        st.warning("Upgrade to Premium to unlock advanced features!")
        st.stop()

    # Reset button
    if st.button("🔄 Reset App"):
        st.session_state.clear()
        st.experimental_rerun()

    # Placeholders for status messages
    overall_status = st.empty()
    prophet_status = st.empty()
    arima_status = st.empty()
    xgb_status = st.empty()
    automl_status = st.empty()

    uploaded_file = st.file_uploader("Upload your sales data file", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(data)

            st.markdown(
                """
                <div style="text-align: center;">
                    <h2 style="color: #2B3A42;">🛠️ Map Your Columns</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                date_column = st.selectbox(
                    "📅 Select the Date Column:",
                    ["-- Select Column --"] + list(data.columns),
                    key="date_col"
                )
            with col2:
                sales_column = st.selectbox(
                    "💰 Select the Sales Column:",
                    ["-- Select Column --"] + list(data.columns),
                    key="sales_col"
                )
            with col3:
                if date_column == "-- Select Column --" or sales_column == "-- Select Column --":
                    st.warning("Please select the Date and Sales columns first.")
                    category_columns = None
                else:
                    category_columns = st.multiselect(
                        "🏷️ Select Category Columns (Optional):",
                        options=[col for col in data.columns if col not in [date_column, sales_column]],
                        key="category_cols"
                    )

            if date_column != "-- Select Column --":
                data[date_column] = pd.to_datetime(data[date_column], errors="coerce")

            # Add Time Budget Slider
            st.markdown("### ⏱️ AutoML Time Budget")
            time_budget = st.slider(
                "Set the time budget for AutoML training (in seconds):",
                min_value=60,
                max_value=1200,
                value=300,
                step=60,
                help="Increase the time budget for larger datasets or more complex models."
            )

            # Scenario Planning on Sidebar
            if date_column != "-- Select Column --" and sales_column != "-- Select Column --":
                st.sidebar.markdown("### 🎯 Scenario Planning")
                demand_shock = st.sidebar.slider(
                    "Simulate Demand Shock (% Change in Sales):",
                    min_value=-50, max_value=50, value=0, step=5
                )
                seasonality_adjustment = st.sidebar.slider(
                    "Adjust Seasonality Strength (% Change):",
                    min_value=-50, max_value=50, value=0, step=5
                )
                external_shock = st.sidebar.checkbox("Simulate External Shock (e.g., Economic Downturn)")

                if category_columns:
                    st.sidebar.markdown("### 🎯 Scenario Planning by Category")
                    category_adjustments = []
                    category_date_ranges = {}
                    for i, category_column in enumerate(category_columns):
                        st.sidebar.markdown(f"#### {category_column} Adjustments")
                        adjustment = st.sidebar.slider(
                            f"Adjust Sales for {category_column} (% Change):",
                            min_value=-50,
                            max_value=50,
                            value=10,
                            step=5,
                            key=f"category_adjustment_{i}"
                        )
                        category_adjustments.append(adjustment)
                        start_date = st.sidebar.date_input(
                            f"Start Date for {category_column}",
                            value=data[date_column].min().to_pydatetime(),
                            key=f"start_date_{i}"
                        )
                        end_date = st.sidebar.date_input(
                            f"End Date for {category_column}",
                            value=data[date_column].max().to_pydatetime(),
                            key=f"end_date_{i}"
                        )
                        if end_date < start_date:
                            st.sidebar.error(f"End date must be after start date for {category_column}.")
                        category_date_ranges[category_column] = (start_date, end_date)
                else:
                    st.sidebar.markdown("ℹ️ No category columns selected. Category-based scenario planning is disabled.")
                    category_adjustments = None
                    category_date_ranges = None
            else:
                st.sidebar.warning("Please select the Date and Sales columns to enable scenario planning.")

            if date_column != "-- Select Column --" and sales_column != "-- Select Column --":
                start_forecast = st.button("✅ Start Forecast", key="start_btn", help="Click to generate your AI-powered forecast")
            else:
                start_forecast = st.button("⏳ Select Columns First", disabled=True, key="start_disabled")

            # ------------------- MAIN FORECAST LOGIC ------------------- #
            if start_forecast:
                # 1) Preprocessing
                with st.spinner("🔍 Preprocessing data..."):
                    processed_data, last_historical_value, y_original = preprocess_data(
                        data, date_column, sales_column, category_columns
                    )
                    time.sleep(1)
                if processed_data is None:
                    st.error("Preprocessing failed. Please check your data.")
                    return
                st.success("✅ Data Preprocessed Successfully!")
                last_historical_date = y_original["ds"].max()
                overall_status.info(f"🔍 Last Historical Date: {last_historical_date}")
                time.sleep(0.5)

                # 2) Apply scenarios
                with st.spinner("🔍 Applying scenarios to training data..."):
                    scenario_data = apply_scenarios(
                        processed_data.copy(),
                        demand_shock,
                        seasonality_adjustment,
                        external_shock,
                        category_columns,
                        category_adjustments,
                        category_date_ranges
                    )
                    time.sleep(1)
                st.success("✅ Scenarios Applied!")
                time.sleep(0.5)

                st.markdown(
                    """
                    <div style="text-align: center;">
                        <h2 style="color: #2B3A42; font-size: 1.8em;">📅 Preprocessed Monthly Data</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("📊 View Processed Data"):
                    st.dataframe(scenario_data.style.set_properties(**{"text-align": "center"}),
                                 width=1400, height=450)

                # 3) Split data
                testing_period = int(len(scenario_data) * 0.2)
                train = scenario_data.iloc[:-testing_period]
                test = scenario_data.iloc[-testing_period:]
                forecast_period = 24

                overall_status.info("🚀 Starting forecasting process...")
                time.sleep(1)

                # 4) Find best Prophet hyperparameters
                with st.spinner("🚀 Finding the best Prophet hyperparameters..."):
                    best_params, best_rmse = find_best_prophet_params(train)
                    time.sleep(1)
                if best_params is None:
                    st.error("No valid Prophet parameters were found.")
                    return
                st.success(f"✅ Best Prophet Params: {best_params}")
                overall_status.write(f"📉 Best RMSE (CV): {best_rmse:.2f}")
                time.sleep(1)

                # 5) Train Prophet Model
                with st.spinner("🚀 Training Prophet Model..."):
                    prophet_model_name, prophet_res = train_prophet_model(
                        train, test, forecast_period, best_params,
                        last_historical_value, y_original
                    )
                    time.sleep(1)
                prophet_status.success("✅ Prophet Model Training Complete!")

                # 6) Train ARIMA Model
                with st.spinner("🚀 Training ARIMA Model..."):
                    arima_model_name, arima_res = train_arima_model(
                        train, test, forecast_period,
                        last_historical_value, y_original
                    )
                    time.sleep(1)
                arima_status.success("✅ ARIMA Model Training Complete!")

                # 7) Train XGBoost Model
                with st.spinner("🚀 Training XGBoost Model..."):
                    xgb_model_name, xgb_res = train_xgb_model(
                        train, test, forecast_period,
                        last_historical_value, y_original
                    )
                    time.sleep(1)
                xgb_status.success("✅ XGBoost Model Training Complete!")

                # 8) Train AutoML Model
                with st.spinner("🚀 Training AutoML Model..."):
                    automl_model_name, automl_res = train_automl_model(
                        train, test, forecast_period,
                        last_historical_value, y_original, time_budget
                    )
                    time.sleep(1)
                automl_status.success("✅ AutoML Model Training Complete!")

                # 9) Compile results
                st.success("🎉 Forecasting process completed!")
                time.sleep(1)
                results = {
                    prophet_model_name: prophet_res,
                    arima_model_name: arima_res,
                    xgb_model_name: xgb_res,
                    automl_model_name: automl_res
                }

                # Store model results in session state
                st.session_state.model_results = results

                # 10) Display numerical performance comparison
                st.subheader("📌 Model Performance Comparison (Numerical)")
                comparison_data = []
                for model, res in results.items():
                    if isinstance(res, dict) and "RMSE" in res and "MAPE" in res:
                        comparison_data.append({
                            "Model": model,
                            "RMSE": float(res["RMSE"]),
                            "MAPE": float(res["MAPE"])
                        })
                    else:
                        st.warning(f"Invalid result format for {model}.")
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data).sort_values(by="RMSE")
                    st.dataframe(comparison_df.style.highlight_min(subset=["RMSE", "MAPE"], color="lightgreen"))
                else:
                    st.error("No valid model results available for numerical comparison.")

                # 11) Compute Shape Score and Combined Score for model selection
                # Let the user adjust weights via sidebar
                st.sidebar.markdown("### ⚖️ Model Selection Weights")
                alpha = st.sidebar.slider("Weight for RMSE", 0.0, 1.0, 0.5)
                beta = st.sidebar.slider("Weight for Shape Fit (Correlation)", 0.0, 1.0, 0.5)

                # Check if model results are already computed
                if st.session_state.model_results:
                    # Compute shape scores and combined scores using the updated alpha and beta
                    for model, res in st.session_state.model_results.items():
                        forecast_df = res["Forecast"]
                        # Align test data and forecast predictions
                        match_len = min(len(test["y"]), len(forecast_df))
                        actual = test["y"].iloc[:match_len].values
                        pred = forecast_df["yhat"].iloc[:match_len].values
                        corr = shape_score(actual, pred)
                        res["Shape"] = corr

                    # Compute combined scores
                    max_rmse = max(res["RMSE"] for res in st.session_state.model_results.values())
                    for model, res in st.session_state.model_results.items():
                        res["Combined"] = combined_score(res["RMSE"], res["Shape"], max_rmse, alpha, beta)

                    # Create a new comparison dataframe including shape and combined scores
                    combined_data = []
                    for model, res in st.session_state.model_results.items():
                        combined_data.append({
                            "Model": model,
                            "RMSE": res["RMSE"],
                            "MAPE": res["MAPE"],
                            "Shape (corr)": res["Shape"],
                            "Combined Score": res["Combined"]
                        })
                    combined_df = pd.DataFrame(combined_data).sort_values(by="Combined Score")

                    # Display the updated comparison
                    st.subheader("📌 Model Performance Comparison (Weighted)")
                    st.dataframe(combined_df.style.highlight_min(subset=["Combined Score"], color="lightgreen"))

                    best_model = combined_df.iloc[0]["Model"]
                    st.success(f"✨ **AI-Selected Best Model (Combined):** {best_model}")
                else:
                    st.warning("No model results found. Please train the models first.")

                # 12) Plot forecast (e.g., Multi-Model Forecast Visualization)
                st.markdown("### 🔍 Forecast Comparison Across Models")
                model_colors = {
                    "Prophet": "blue",
                    "ARIMA": "green",
                    "XGBoost": "red",
                    "AutoML": "purple"
                }
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_original["ds"],
                    y=y_original["y_original"],
                    mode="lines",
                    name="Historical Data",
                    line=dict(color="black", width=2)
                ))
                for model_name, res in results.items():
                    if "Forecast" in res and res["Forecast"] is not None and not res["Forecast"].empty:
                        forecast_df = res["Forecast"]
                        fig.add_trace(go.Scatter(
                            x=forecast_df["ds"],
                            y=forecast_df["yhat"],
                            mode="lines",
                            name=f"{model_name} Forecast",
                            line=dict(width=2, color=model_colors.get(model_name, "gray"))
                        ))
                fig.update_layout(
                    title="📊 Multi-Model Sales Forecast",
                    xaxis_title="Date",
                    yaxis_title="Sales",
                    legend_title="Models",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

                # 13) Download forecast data
                st.markdown("### 📥 Download Forecast Data")
                try:
                    csv = results[best_model]["Forecast"].to_csv(index=False)
                    st.download_button(
                        label="📩 Download Best Model Forecast (CSV)",
                        data=csv,
                        file_name="forecast.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error generating download file: {e}")

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()