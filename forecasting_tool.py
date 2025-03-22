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
            if not isinstance(category_columns, list):
                category_columns = [category_columns]
            grouping_cols = ["year_month"] + category_columns
        else:
            grouping_cols = ["year_month"]

        # 5. Aggregate (sum) sales at the monthly level
        data = data.groupby(grouping_cols, as_index=False)["y"].sum()

        # 6. Convert 'year_month' back to a proper datetime (start-of-month)
        data["ds"] = pd.to_datetime(data["year_month"].dt.to_timestamp(how="start").dt.strftime("%Y-%m"),
                                    format="%Y-%m")
        data.drop(columns=["year_month"], inplace=True)

        # 7. Remove duplicates if any
        subset_cols = ["ds"] + (category_columns if category_columns else [])
        if data.duplicated(subset=subset_cols).any():
            st.warning("⚠️ Duplicate date/category combinations found. Removing duplicates...")
            data = data.drop_duplicates(subset=subset_cols, keep="last")

        # 8. Original values for inspection
        y_original = data[["ds", "y"]].copy().rename(columns={"y": "y_original"})

        # 9. Check stationarity
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

            # Visualization
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
                xaxis_title="Date (YYYY-MM)",
                yaxis_title="Sales",
                template="plotly_white",
                xaxis_tickformat="%Y-%m"
            )
            st.plotly_chart(fig, use_container_width=True)

            differenced_data = data.dropna(subset=["y_diff"]).drop(columns=["y"])
            differenced_data = differenced_data.rename(columns={"y_diff": "y"})
            differenced_data = differenced_data.reset_index(drop=True)
            differenced_data.columns = differenced_data.columns.astype(str)

            return differenced_data[["ds", "y"]], last_historical_value, y_original

        else:
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["ds"],
                y=data["y"],
                mode="lines",
                name="Original Series (Monthly)",
                line=dict(color="blue", width=2)
            ))
            fig.update_layout(
                title="Original Series (Monthly)",
                xaxis_title="Date (YYYY-MM)",
                yaxis_title="Sales",
                template="plotly_white",
                xaxis_tickformat="%Y-%m"
            )
            st.plotly_chart(fig, use_container_width=True)

            data = data.reset_index(drop=True)
            return data[["ds", "y"]], None, y_original

    except Exception as e:
        st.error(f"An error occurred during preprocessing: {e}")
        st.error(f"Debug Info: Columns in data - {data.columns}, Data Shape - {data.shape}")
        return None, None, None

def inverse_difference(forecast_data, first_value):
    if first_value is not None:
        if not isinstance(first_value, (int, float)):
            raise ValueError("first_value must be numeric.")
        if isinstance(forecast_data, pd.Series):
            forecast_data = forecast_data.to_frame(name="yhat")

        if "yhat" in forecast_data.columns:
            forecast_data["yhat"] = first_value + forecast_data["yhat"].cumsum()
        
        if "yhat_upper" in forecast_data.columns:
            forecast_data["yhat_upper"] = first_value + forecast_data["yhat_upper"].cumsum()
        
        if "yhat_lower" in forecast_data.columns:
            forecast_data["yhat_lower"] = first_value + forecast_data["yhat_lower"].cumsum()
    return forecast_data

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
    
    grid = list(ParameterGrid(param_grid))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(evaluate_prophet_params, params, train, initial, horizon, period): params
            for params in grid
        }
        for future in concurrent.futures.as_completed(futures):
            params, rmse = future.result()
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
    return best_params, best_rmse

def apply_scenarios(data, demand_shock, seasonality_adjustment, external_shock, category_scenarios=None):
    """
    Apply scenario adjustments to the data.
    
    Global adjustments (demand shock, seasonality, external shock) are applied first.
    Then, if provided, category-specific adjustments from the dynamic dictionary 'category_scenarios'
    are applied. This dictionary is expected to have the following structure:
    
    {
      "column_name1": {
           "category_value1": {"adjustment": 10, "start_date": <date>, "end_date": <date>},
           "category_value2": {"adjustment": -20, "start_date": <date>, "end_date": <date>},
           ...
      },
      "column_name2": { ... }
    }
    
    Returns:
        The modified DataFrame.
    """
    try:
        if data is None or data.empty:
            st.error("❌ No data provided. Please check your input.")
            return data

        if "ds" not in data.columns or "y" not in data.columns:
            st.error("❌ Required columns 'ds' (date) and 'y' (sales) are missing.")
            return data

        # Global demand shock
        if demand_shock != 0:
            data["y"] = data["y"] * (1 + demand_shock / 100)
            st.write(f"✅ Applied global demand shock: {demand_shock}%")

        # Global seasonality adjustment
        if seasonality_adjustment != 0:
            data["month"] = data["ds"].dt.month
            seasonality_multiplier = 1 + seasonality_adjustment / 100
            data["y"] = data["y"] * (1 + (data["month"] - 1) * (seasonality_multiplier - 1) / 12)
            st.write(f"✅ Applied global seasonality adjustment: {seasonality_adjustment}%")

        # Global external shock
        if external_shock:
            data["y"] = data["y"] * 0.8  # 20% reduction in sales
            st.write("✅ Applied global external shock: 20% reduction in sales")

        # Category-specific adjustments using the dynamic dictionary
        if category_scenarios:
            st.write("🔍 Applying category-specific adjustments...")
            for col, cat_dict in category_scenarios.items():
                if col not in data.columns:
                    st.warning(f"⚠️ Column '{col}' not found in the dataset. Skipping adjustments for this column.")
                    continue

                # Process each category value in the current column
                for cat_val, details in cat_dict.items():
                    adjustment = details.get("adjustment", 0)
                    start_date = pd.to_datetime(details.get("start_date"))
                    end_date = pd.to_datetime(details.get("end_date"))

                    # Filter rows that fall within the specified date range and match the category value
                    mask = (data["ds"] >= start_date) & (data["ds"] <= end_date) & (data[col] == cat_val)
                    if mask.sum() == 0:
                        st.write(f"ℹ️ No rows found for '{cat_val}' in '{col}' between {start_date.date()} and {end_date.date()}.")
                        continue

                    # Apply adjustment
                    data.loc[mask, "y"] *= (1 + adjustment / 100)
                    st.write(
                        f"✅ Applied {adjustment}% adjustment to '{cat_val}' in '{col}' "
                        f"(from {start_date.date()} to {end_date.date()})"
                    )
        else:
            st.warning("⚠️ No category-specific adjustments provided. Skipping this step.")

        return data

    except Exception as e:
        st.error(f"❌ An error occurred while applying scenarios: {e}")
        st.error(f"Debug Info: Columns in data - {data.columns}, Data Shape - {data.shape}")
        return data

def train_prophet_model(train, test, forecast_period, best_params, last_historical_value, y_original):
    result = {}
    try:
        st.write("DEBUG: Prophet - last training date:", train["ds"].iloc[-1])
        model = Prophet(
            seasonality_mode=best_params["seasonality_mode"],
            changepoint_prior_scale=best_params["changepoint_prior_scale"]
        )
        try:
            model = detect_and_add_seasonalities(model, train)
        except Exception as e:
            st.warning(f"Seasonality detection failed: {e}. Proceeding without additional seasonalities.")
        model.fit(train)
        
        # Use MS for start-of-month alignment
        future = model.make_future_dataframe(periods=forecast_period, freq="MS", include_history=False)
        st.write("DEBUG: Prophet - first forecast date (unfiltered):", future["ds"].iloc[0])
        forecast = model.predict(future)
        forecast = forecast[forecast["ds"] > train["ds"].max()]
        
        st.write("DEBUG: Prophet - first forecast date (filtered):", forecast["ds"].iloc[0])
        st.write("DEBUG: Prophet - last forecast date:", forecast["ds"].iloc[-1])
        
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
        st.write("DEBUG: ARIMA - last training date:", train["ds"].iloc[-1])
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
            trace=False,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True
        )
        arima_forecast = model.predict(n_periods=forecast_period)
        if len(arima_forecast) < forecast_period:
            forecast_period = len(arima_forecast)
        
        # Use MS for start-of-month alignment
        forecast_dates = pd.date_range(
            start=train["ds"].iloc[-1] + pd.DateOffset(months=1),
            periods=forecast_period,
            freq="MS"
        )
        st.write("DEBUG: ARIMA - forecast start date:", forecast_dates[0])
        st.write("DEBUG: ARIMA - forecast end date:", forecast_dates[-1])
        
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
        st.write("DEBUG: XGBoost - last training date:", train["ds"].iloc[-1])
        
        max_lag = min(12, len(train) - 1)
        rolling_windows = [3, 6] if len(train) > 6 else [3]
        data_xgb = train.copy()
        
        # Create lag features
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
        
        # Iterative forecasting
        future_forecasts = []
        last_row = data_xgb.iloc[-1].copy()
        last_date = train["ds"].iloc[-1]
        
        for i in range(forecast_period):
            future_date = last_date + pd.DateOffset(months=i+1)
            future_row = {}
            for col in feature_cols:
                future_row[col] = last_row[col]
            
            # Update time-based features for the new future_date
            future_row["month"] = future_date.month
            future_row["quarter"] = future_date.quarter
            future_row["year"] = future_date.year
            future_row["sin_month"] = np.sin(2 * np.pi * future_row["month"] / 12)
            future_row["cos_month"] = np.cos(2 * np.pi * future_row["month"] / 12)
            
            X_future = pd.DataFrame([future_row])
            pred = model.predict(X_future)[0]
            future_forecasts.append(pred)
            
            # Shift lag features
            for lag in range(max_lag, 1, -1):
                last_row[f"lag_{lag}"] = last_row[f"lag_{lag-1}"]
            last_row["lag_1"] = pred
            # Rolling features remain unchanged for simplicity
        
        # Construct forecast DataFrame with freq="MS"
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_period,
            freq="MS"
        )
        st.write("DEBUG: XGBoost - forecast start date:", forecast_dates[0])
        st.write("DEBUG: XGBoost - forecast end date:", forecast_dates[-1])
        
        forecast_df = pd.DataFrame({"ds": forecast_dates, "yhat": future_forecasts})
        
        if isinstance(last_historical_value, (int, float)):
            forecast_df["yhat"] = inverse_difference(forecast_df["yhat"], last_historical_value)
        
        matching_length = min(len(test["y"]), len(forecast_df))
        rmse = mean_squared_error(test["y"].iloc[:matching_length], forecast_df["yhat"].iloc[:matching_length]) ** 0.5
        mape = mean_absolute_percentage_error(test["y"].iloc[:matching_length], forecast_df["yhat"].iloc[:matching_length])
        
        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast_df}
    except Exception as e:
        st.warning(f"XGBoost Model failed: {e}")
    return ("XGBoost", result)

def train_automl_model(train, test, forecast_period, last_historical_value, y_original, time_budget=None):
    result = {}
    try:
        st.write("DEBUG: AutoML - last training date:", train["ds"].iloc[-1])

        data_automl = train.copy()
        max_lag = min(24, len(train) - 1)
        if len(train) <= 6:
            max_lag = min(3, len(train) - 1)
        elif len(train) <= 12:
            max_lag = min(6, len(train) - 1)
        elif len(train) <= 24:
            max_lag = min(12, len(train) - 1)

        for lag in range(1, max_lag + 1):
            data_automl[f"lag_{lag}"] = data_automl["y"].shift(lag)

        for window in [3, 6, 12]:
            data_automl[f"rolling_mean_{window}"] = data_automl["y"].rolling(window=window, min_periods=1).mean()
            data_automl[f"rolling_std_{window}"] = data_automl["y"].rolling(window=window, min_periods=1).std()

        if len(train) > 12:
            data_automl["yoy_growth"] = (data_automl["y"] / data_automl["y"].shift(12)) - 1
        else:
            data_automl["yoy_growth"] = 0

        data_automl["y_diff"] = data_automl["y"].diff().fillna(0)
        data_automl["rolling_mean_growth"] = data_automl["y"].rolling(window=3).mean().diff().fillna(0)
        data_automl["sin_month"] = np.sin(2 * np.pi * data_automl["ds"].dt.month / 12)
        data_automl["cos_month"] = np.cos(2 * np.pi * data_automl["ds"].dt.month / 12)

        if data_automl["y"].min() > 0 and (data_automl["y"].max() / data_automl["y"].min() > 5):
            data_automl["y_log"] = np.log1p(data_automl["y"])
            apply_log = True
        else:
            data_automl["y_log"] = data_automl["y"]
            apply_log = False

        data_automl.dropna(inplace=True)
        feature_cols = [col for col in data_automl.columns if col not in ["y", "ds", "y_log"]]
        y_train = data_automl["y_log"] if apply_log else data_automl["y"]
        X_train = data_automl[feature_cols]

        if time_budget is None:
            time_budget = min(600, max(60, len(train) * 0.1 + len(feature_cols) * 2))
            st.info(f"Dynamic time budget set to {time_budget} seconds.")

        automl_model = AutoML()
        automl_model.fit(
            X_train=X_train,
            y_train=y_train,
            task="regression",
            time_budget=time_budget,
            eval_method="cv",
            estimator_list=["xgboost", "lgbm", "rf", "catboost"],
            metric="r2",
            early_stop=True,
            verbose=1
        )

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
                    future_row[f"rolling_mean_{window}"] = last_row[f"rolling_mean_{window}"] + (
                        last_row["y_log"] - last_row[f"lag_{window}"]
                    ) / window
                else:
                    future_row[f"rolling_mean_{window}"] = last_row[f"rolling_mean_{window}"] + (
                        last_row["y"] - last_row[f"lag_{window}"]
                    ) / window
                future_row[f"rolling_std_{window}"] = last_row[f"rolling_std_{window}"]
            future_row["yoy_growth"] = last_row["yoy_growth"]
            future_row["y_diff"] = last_row["y_diff"]
            future_row["rolling_mean_growth"] = last_row["rolling_mean_growth"]
            future_row["sin_month"] = np.sin(2 * np.pi * (last_row["ds"].month + i) / 12)
            future_row["cos_month"] = np.cos(2 * np.pi * (last_row["ds"].month + i) / 12)
            future_features.append(future_row)
            for key, value in future_row.items():
                last_row[key] = value

        future_df = pd.DataFrame(future_features)
        for col in X_train.columns:
            if col not in future_df.columns:
                future_df[col] = 0
        future_df = future_df[X_train.columns]

        automl_forecast = automl_model.predict(future_df)
        if apply_log:
            automl_forecast = np.expm1(automl_forecast)

        # Use MS for start-of-month alignment
        forecast_dates = pd.date_range(
            start=train["ds"].iloc[-1] + pd.DateOffset(months=1),
            periods=forecast_period,
            freq="MS"
        )
        st.write("DEBUG: AutoML - forecast start date:", forecast_dates[0])
        st.write("DEBUG: AutoML - forecast end date:", forecast_dates[-1])

        forecast_df = pd.DataFrame({
            "ds": forecast_dates,
            "yhat": automl_forecast,
            "yhat_lower": automl_forecast * 0.9,
            "yhat_upper": automl_forecast * 1.1
        })

        if isinstance(last_historical_value, (int, float)):
            forecast_df["yhat"] = inverse_difference(forecast_df["yhat"], last_historical_value)
            forecast_df["yhat_lower"] = inverse_difference(forecast_df["yhat_lower"], last_historical_value)
            forecast_df["yhat_upper"] = inverse_difference(forecast_df["yhat_upper"], last_historical_value)

        matching_length = min(len(test["y"]), len(forecast_df))
        rmse = np.sqrt(mean_squared_error(test["y"].values, forecast_df["yhat"][:len(test["y"])]))
        mape = mean_absolute_percentage_error(test["y"].values, forecast_df["yhat"][:len(test["y"])])

        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast_df}

    except Exception as e:
        st.error(f"AutoML Model failed: {e}")
    return ("AutoML", result)

def shape_score(actual, forecast):
    if len(actual) != len(forecast):
        min_len = min(len(actual), len(forecast))
        actual = actual[:min_len]
        forecast = forecast[:min_len]
    corr, _ = pearsonr(actual, forecast)
    return corr

def combined_score(rmse, corr, max_rmse, alpha, beta):
    norm_rmse = rmse / max_rmse
    return alpha * norm_rmse + beta * (1 - corr)

def main():
    user_id = "user123"
    subscription_level = "premium"
    if subscription_level != "premium":
        st.warning("Upgrade to Premium to unlock advanced features!")
        st.stop()

    if st.button("🔄 Reset App"):
        st.session_state.clear()
        st.experimental_rerun()

    overall_status = st.empty()
    prophet_status = st.empty()
    arima_status = st.empty()
    xgb_status = st.empty()
    automl_status = st.empty()

    total_steps = 12
    progress_bar = st.progress(0)
    step_message = st.empty()

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

            st.markdown("### ⏱️ AutoML Time Budget")
            time_budget = st.slider(
                "Set the time budget for AutoML training (in seconds):",
                min_value=60,
                max_value=1200,
                value=300,
                step=60,
                help="Increase the time budget for larger datasets or more complex models."
            )

            # Define global scenario planning variables
            if date_column != "-- Select Column --" and sales_column != "-- Select Column --":
                st.sidebar.markdown("### 🎯 Scenario Planning")
                demand_shock = st.sidebar.slider(
                    "Simulate Demand Shock (% Change in Sales):",
                    min_value=-50,
                    max_value=50,
                    value=0,
                    step=5
                )
                seasonality_adjustment = st.sidebar.slider(
                    "Adjust Seasonality Strength (% Change):",
                    min_value=-50,
                    max_value=50,
                    value=0,
                    step=5
                )
                external_shock = st.sidebar.checkbox("Simulate External Shock (e.g., Economic Downturn)")

                # Build dynamic category adjustments dictionary
                category_scenarios = {}

                if category_columns:
                    st.sidebar.markdown("### 🎯 Scenario Planning by Category (Dynamic)")
                    
                    for col in category_columns:
                        st.sidebar.markdown(f"#### Adjustments for '{col}'")
                        
                        # 1) Get all unique categories in the current column
                        unique_cats = sorted(data[col].dropna().unique())

                        # 2) Let the user pick which categories they want to adjust
                        selected_cats = st.sidebar.multiselect(
                            f"Pick categories in '{col}' to adjust:",
                            options=unique_cats,
                            help=f"Select one or more categories from '{col}' that you want to apply adjustments to."
                        )

                        # Initialize a dictionary for this column
                        category_scenarios[col] = {}

                        # 3) Only create sliders/date inputs for the selected categories
                        for cat in selected_cats:
                            with st.sidebar.expander(f"Adjust '{cat}' in '{col}'"):
                                cat_adjust = st.slider(
                                    f"Percentage change for '{cat}'",
                                    min_value=-50,
                                    max_value=50,
                                    value=0,
                                    step=5,
                                    help=f"Adjust sales for category '{cat}' within column '{col}'"
                                )
                                cat_start = st.date_input(
                                    f"Start date for '{cat}'",
                                    value=data[date_column].min().to_pydatetime()
                                )
                                cat_end = st.date_input(
                                    f"End date for '{cat}'",
                                    value=data[date_column].max().to_pydatetime()
                                )

                                # Store the user's inputs in the dictionary
                                category_scenarios[col][cat] = {
                                    "adjustment": cat_adjust,
                                    "start_date": cat_start,
                                    "end_date": cat_end
                                }
                else:
                    st.sidebar.markdown("ℹ️ No category columns selected. Category-based scenario planning is disabled.")

            else:
                st.sidebar.warning("Please select the Date and Sales columns to enable scenario planning.")
                # Set default values to prevent undefined errors
                demand_shock = 0
                seasonality_adjustment = 0
                external_shock = False
                category_scenarios = {}

            if date_column != "-- Select Column --" and sales_column != "-- Select Column --":
                start_forecast = st.button("✅ Start Forecast", key="start_btn",
                                           help="Click to generate your AI-powered forecast")
            else:
                start_forecast = st.button("⏳ Select Columns First", disabled=True, key="start_disabled")

            if start_forecast:
                step = 1
                step_message.text(f"Step {step} of {total_steps}: Preprocessing data...")
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
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(0.5)

                step += 1
                step_message.text(f"Step {step} of {total_steps}: Applying business scenarios...")
                with st.spinner("🔍 Applying scenarios to training data..."):
                    scenario_data = apply_scenarios(
                        data=processed_data.copy(),
                        demand_shock=demand_shock,
                        seasonality_adjustment=seasonality_adjustment,
                        external_shock=external_shock,
                        category_scenarios=category_scenarios
                    )
                    time.sleep(1)
                st.success("✅ Scenarios Applied!")
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(0.5)

                step += 1
                step_message.text(f"Step {step} of {total_steps}: Reviewing processed data...")
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
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(0.5)

                step += 1
                step_message.text(f"Step {step} of {total_steps}: Splitting data into training and test sets...")
                testing_period = int(len(scenario_data) * 0.2)
                train = scenario_data.iloc[:-testing_period]
                test = scenario_data.iloc[-testing_period:]
                forecast_period = 24

                # Debug: show last date of train, first date of test
                st.write("DEBUG: Train last date:", train["ds"].iloc[-1])
                st.write("DEBUG: Test first date:", test["ds"].iloc[0])

                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(1)

                step += 1
                step_message.text(f"Step {step} of {total_steps}: Tuning Prophet model...")
                with st.spinner("🚀 Finding the best Prophet hyperparameters..."):
                    best_params, best_rmse = find_best_prophet_params(train)
                    time.sleep(1)
                if best_params is None:
                    st.error("No valid Prophet parameters were found.")
                    return
                st.success(f"✅ Best Prophet Params: {best_params}")
                overall_status.write(f"📉 Best RMSE (CV): {best_rmse:.2f}")
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(1)

                step += 1
                step_message.text(f"Step {step} of {total_steps}: Training Prophet model...")
                with st.spinner("🚀 Training Prophet Model..."):
                    prophet_model_name, prophet_res = train_prophet_model(
                        train, test, forecast_period, best_params,
                        last_historical_value, y_original
                    )
                    time.sleep(1)
                prophet_status.success("✅ Prophet Model Training Complete!")
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(1)

                step += 1
                step_message.text(f"Step {step} of {total_steps}: Training ARIMA model...")
                with st.spinner("🚀 Training ARIMA Model..."):
                    arima_model_name, arima_res = train_arima_model(
                        train, test, forecast_period,
                        last_historical_value, y_original
                    )
                    time.sleep(1)
                arima_status.success("✅ ARIMA Model Training Complete!")
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(1)

                step += 1
                step_message.text(f"Step {step} of {total_steps}: Training XGBoost model...")
                with st.spinner("🚀 Training XGBoost Model..."):
                    xgb_model_name, xgb_res = train_xgb_model(
                        train, test, forecast_period,
                        last_historical_value, y_original
                    )
                    time.sleep(1)
                xgb_status.success("✅ XGBoost Model Training Complete!")
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(1)

                step += 1
                step_message.text(f"Step {step} of {total_steps}: Training AutoML model...")
                with st.spinner("🚀 Training AutoML Model..."):
                    automl_model_name, automl_res = train_automl_model(
                        train, test, forecast_period,
                        last_historical_value, y_original, time_budget
                    )
                    time.sleep(1)
                automl_status.success("✅ AutoML Model Training Complete!")
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(1)

                step += 1
                step_message.text(f"Step {step} of {total_steps}: Compiling forecast results...")
                st.success("🎉 Forecasting process completed!")
                time.sleep(1)
                results = {
                    prophet_model_name: prophet_res,
                    arima_model_name: arima_res,
                    xgb_model_name: xgb_res,
                    automl_model_name: automl_res
                }
                st.session_state.model_results = results
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(1)

                step += 1
                step_message.text(f"Step {step} of {total_steps}: Displaying model performance comparison...")
                if st.session_state.model_results:
                    for model, res in st.session_state.model_results.items():
                        forecast_df = res["Forecast"]
                        match_len = min(len(test["y"]), len(forecast_df))
                        actual = test["y"].iloc[:match_len].values
                        pred = forecast_df["yhat"].iloc[:match_len].values
                        corr = shape_score(actual, pred)
                        res["Shape (corr)"] = corr

                    max_rmse = max(res["RMSE"] for res in st.session_state.model_results.values())
                    for model, res in st.session_state.model_results.items():
                        res["Combined Score"] = combined_score(res["RMSE"], res["Shape (corr)"], max_rmse, 0.5, 1.0)

                    comparison_data = []
                    for model, res in st.session_state.model_results.items():
                        comparison_data.append({
                            "Model": model,
                            "RMSE": res["RMSE"],
                            "MAPE": res["MAPE"],
                            "Shape (corr)": res["Shape (corr)"],
                            "Combined Score": res["Combined Score"]
                        })
                    comparison_df = pd.DataFrame(comparison_data).sort_values(by="Combined Score")
                    st.dataframe(comparison_df.style.highlight_min(subset=["Combined Score"], color="lightgreen"))
                    best_model = comparison_df.iloc[0]["Model"]
                    st.success(f"✨ **AI-Selected Best Model (Combined):** {best_model}")
                else:
                    st.warning("No model results found. Please train the models first.")
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(1)

                step += 1
                step_message.text(f"Step {step} of {total_steps}: Finalizing forecast visualization...")
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
                    template="plotly_white",
                    xaxis_tickformat="%Y-%m"
                )
                st.plotly_chart(fig, use_container_width=True)
                progress_bar.progress(100)
                step_message.text("All steps completed!")

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
