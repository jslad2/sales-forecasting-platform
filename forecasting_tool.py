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
import calendar

# Enable Wide Mode (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(layout="wide", page_title="Time Series Forecasting", page_icon="📈")

# Add dark mode toggle
if "theme" not in st.session_state:
    st.session_state.theme = "light"

theme = st.radio("🌙 Theme Mode:", ["Light", "Dark"], index=0 if st.session_state.theme=="light" else 1)
st.session_state.theme = theme

if st.session_state.theme=="dark":
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
        """, unsafe_allow_html=True)
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
        """, unsafe_allow_html=True)

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
    Preprocess the uploaded data for Prophet.
    Returns:
      - data: Processed DataFrame with columns ds, y (if non-stationary, y is differenced),
              and category columns (if provided)
      - last_historical_value: The last y value (if differencing was applied), otherwise None
      - y_original: A copy of the processed data for inspection
      - is_diff: Boolean indicating whether differencing was applied
      Also creates a re-aggregated DataFrame for a clean monthly chart.
    """
    try:
        # 1. Convert date column to datetime and drop missing rows
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data.dropna(subset=[date_column, sales_column], inplace=True)

        # 2. Rename columns for Prophet
        data = data.rename(columns={date_column: "ds", sales_column: "y"})

        # 3. Create monthly period column
        data["year_month"] = data["ds"].dt.to_period("M")

        # 4. Group by year_month and category columns (if provided)
        if category_columns:
            if not isinstance(category_columns, list):
                category_columns = [category_columns]
            grouping_cols = ["year_month"] + category_columns
        else:
            grouping_cols = ["year_month"]

        data = data.groupby(grouping_cols, as_index=False)["y"].sum()

        # 5. Convert year_month back to datetime (start-of-month)
        data["ds"] = pd.to_datetime(
            data["year_month"].dt.to_timestamp(how="start").dt.strftime("%Y-%m"),
            format="%Y-%m"
        )
        data.drop(columns=["year_month"], inplace=True)

        # 6. Remove duplicates
        if category_columns:
            subset_cols = ["ds", "y"] + category_columns
        else:
            subset_cols = ["ds", "y"]
        if data.duplicated(subset=subset_cols).any():
            st.warning("⚠️ Duplicate date/category combinations found. Removing duplicates...")
            data = data.drop_duplicates(subset=subset_cols, keep="last")

        # 7. Save original processed data for inspection
        if category_columns:
            y_original = data.copy()
        else:
            y_original = data[["ds", "y"]].copy().rename(columns={"y": "y_original"})

        y_original = data.copy()
        if category_columns:
            y_original = y_original.groupby('ds', as_index=False)['y'].sum()
        y_original = y_original.rename(columns={'y': 'y_original'})

        # 8. Re-aggregate data by ds for charting (one line per month)
        re_agg_chart = data.groupby("ds", as_index=False)["y"].sum()

        # 9. Check stationarity on the aggregated series
        stationarity_result = check_stationarity(re_agg_chart["y"])
        st.markdown(
            f"""
            <div style="text-align: center;">
                <h2 style="color: #2B3A42;">📊 Stationarity Test</h2>
                <p style="font-size: 1.2rem;">Conclusion: The series is <strong>{stationarity_result}</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 10. If non-stationary, apply differencing to both the aggregated chart and the model data
        if stationarity_result == "Non-Stationary":
            st.warning("Applying differencing to stabilize the series for charting and modeling.")
            
            # For charting: compute differenced aggregated series
            re_agg_chart["y_diff"] = re_agg_chart["y"].diff()
            last_historical_value = re_agg_chart["y"].iloc[-1]
            is_diff = True

            # For modeling: physically difference the actual data
            data["y"] = data["y"].diff()  # This creates a differenced column
            data = data.dropna().reset_index(drop=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=re_agg_chart["ds"],
                y=re_agg_chart["y"],
                mode="lines",
                name="Original Aggregated Series",
                line=dict(color="blue", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=re_agg_chart["ds"].iloc[1:],
                y=re_agg_chart["y_diff"].dropna(),
                mode="lines",
                name="Differenced Aggregated Series",
                line=dict(color="orange", width=2, dash="dot")
            ))
            fig.update_layout(
                title="Original vs Differenced Series (Monthly, Aggregated)",
                xaxis_title="Date (YYYY-MM)",
                yaxis_title="Sales",
                template="plotly_white",
                xaxis_tickformat="%Y-%m"
            )
            st.plotly_chart(fig, use_container_width=True)
            return data, last_historical_value, y_original, is_diff
        else:
            is_diff = False
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=re_agg_chart["ds"],
                y=re_agg_chart["y"],
                mode="lines",
                name="Original Series (Monthly, Aggregated)",
                line=dict(color="blue", width=2)
            ))
            fig.update_layout(
                title="Original Series (Monthly, Aggregated)",
                xaxis_title="Date (YYYY-MM)",
                yaxis_title="Sales",
                template="plotly_white",
                xaxis_tickformat="%Y-%m"
            )
            st.plotly_chart(fig, use_container_width=True)
            data = data.reset_index(drop=True)
            return data, None, y_original, is_diff

    except Exception as e:
        st.error(f"An error occurred during preprocessing: {e}")
        st.error(f"Debug Info: Columns in data - {data.columns}, Data Shape - {data.shape}")
        return None, None, None, None

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
        param_grid = {"changepoint_prior_scale": [0.01, 0.1], "seasonality_mode": ["additive"]}
    else:
        param_grid = {"changepoint_prior_scale": [0.01, 0.05, 0.1, 0.2, 0.3], "seasonality_mode": ["additive", "multiplicative"]}
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
        futures = {executor.submit(evaluate_prophet_params, params, train, initial, horizon, period): params for params in grid}
        for future in concurrent.futures.as_completed(futures):
            params, rmse = future.result()
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
    return best_params, best_rmse

def adjust_forecast_by_category(forecast_df, category_scenarios):
    # For each scenario in the dynamic dictionary,
    # if the forecast date falls within the specified date range, adjust yhat.
    for col, cat_dict in category_scenarios.items():
        for cat_val, details in cat_dict.items():
            adjustment = details.get("adjustment", 0)
            start_date = pd.to_datetime(details.get("start_date"))
            end_date = pd.to_datetime(details.get("end_date"))
            mask = (forecast_df["ds"] >= start_date) & (forecast_df["ds"] <= end_date)
            if mask.any():
                # Apply category-specific adjustment
                forecast_df.loc[mask, "yhat"] *= (1 + adjustment / 100)
                if "yhat_lower" in forecast_df.columns:
                    forecast_df.loc[mask, "yhat_lower"] *= (1 + adjustment / 100)
                if "yhat_upper" in forecast_df.columns:
                    forecast_df.loc[mask, "yhat_upper"] *= (1 + adjustment / 100)
    
    # Aggregate category-level forecasts to maintain temporal consistency
    if any(col in forecast_df.columns for col in category_scenarios.keys()):
        agg_dict = {
            "yhat": "sum",
            "yhat_lower": "sum",
            "yhat_upper": "sum"
        }
        forecast_df = forecast_df.groupby("ds", as_index=False).agg(agg_dict)
    
    # Ensure temporal order and reset index
    forecast_df = forecast_df.sort_values("ds").reset_index(drop=True)
    
    return forecast_df

def adjust_forecast(forecast_df, demand_shock, seasonality_adjustment, external_shock, category_scenarios=None):
    # Apply global adjustments
    if demand_shock != 0:
        forecast_df["yhat"] *= (1 + demand_shock / 100)
        if "yhat_lower" in forecast_df.columns:
            forecast_df["yhat_lower"] *= (1 + demand_shock / 100)
        if "yhat_upper" in forecast_df.columns:
            forecast_df["yhat_upper"] *= (1 + demand_shock / 100)
    if seasonality_adjustment != 0:
        forecast_df["month"] = forecast_df["ds"].dt.month
        seasonality_multiplier = 1 + seasonality_adjustment / 100
        forecast_df["yhat"] *= (1 + (forecast_df["month"] - 1) * (seasonality_multiplier - 1) / 12)
        if "yhat_lower" in forecast_df.columns:
            forecast_df["yhat_lower"] *= (1 + (forecast_df["month"] - 1) * (seasonality_multiplier - 1) / 12)
        if "yhat_upper" in forecast_df.columns:
            forecast_df["yhat_upper"] *= (1 + (forecast_df["month"] - 1) * (seasonality_multiplier - 1) / 12)
    if external_shock:
        forecast_df["yhat"] *= 0.8
        if "yhat_lower" in forecast_df.columns:
            forecast_df["yhat_lower"] *= 0.8
        if "yhat_upper" in forecast_df.columns:
            forecast_df["yhat_upper"] *= 0.8
    # Then apply category-specific adjustments
    if category_scenarios:
        forecast_df = adjust_forecast_by_category(forecast_df, category_scenarios)
    return forecast_df

def train_prophet_model(train, test, forecast_period, best_params, last_historical_value,
                        is_diff, demand_shock, seasonality_adjustment, external_shock, category_scenarios=None):
    result = {}
    try:
        # If category adjustments are used, aggregate training data by date.
        if category_scenarios:
            train = train.groupby("ds", as_index=False).agg({"y": "sum"})
        
        # Initialize Prophet with tuned parameters.
        model = Prophet(
            seasonality_mode=best_params["seasonality_mode"],
            changepoint_prior_scale=best_params["changepoint_prior_scale"]
        )
        try:
            model = detect_and_add_seasonalities(model, train)
        except Exception as e:
            st.warning(f"Seasonality detection failed: {e}. Proceeding without additional seasonalities.")

        # Fit the model.
        model.fit(train)
        
        # Create a future dataframe.
        future = model.make_future_dataframe(periods=forecast_period, freq="MS", include_history=False)
        forecast = model.predict(future)
        forecast = forecast[forecast["ds"] > train["ds"].max()]

        # Apply scenario adjustments.
        forecast = adjust_forecast(forecast, demand_shock, seasonality_adjustment, external_shock, category_scenarios)

        # Inverse differencing if needed.
        if is_diff and last_historical_value is not None:
            forecast = inverse_difference(forecast, last_historical_value)

        # Evaluate on the overlapping period.
        match_len = min(len(test["y"]), len(forecast))
        rmse = mean_squared_error(test["y"].iloc[:match_len], forecast["yhat"].iloc[:match_len], squared=False)
        mape = mean_absolute_percentage_error(test["y"].iloc[:match_len], forecast["yhat"].iloc[:match_len])

        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast}

    except Exception as e:
        st.warning(f"Prophet Model failed: {e}")

    return "Prophet", result

def train_arima_model(train, test, forecast_period, last_historical_value, is_diff,
                      demand_shock, seasonality_adjustment, external_shock, category_scenarios=None):
    result = {}
    try:
        # If category adjustments are used, aggregate training data by date.
        if category_scenarios:
            train = train.groupby("ds", as_index=False).agg({"y": "sum"})
        
        # Detect seasonality
        try:
            decomposition = seasonal_decompose(train["y"], model="additive", period=12)
            seasonality_present = np.any(np.abs(decomposition.seasonal) > 0.01)
            acf_values = acf(train["y"], nlags=12, fft=False)
            seasonality_confirmed = any(np.abs(acf_values[1:]) > 0.2)
            seasonal = seasonality_present and seasonality_confirmed
        except Exception as e:
            st.warning(f"Error in seasonality analysis: {e}")
            seasonal = False

        # Fit auto_arima
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
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True
        )

        preds = model.predict(n_periods=forecast_period)
        forecast_dates = pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1),
                                       periods=len(preds), freq="MS")
        forecast_df = pd.DataFrame({"ds": forecast_dates, "yhat": preds})

        # Apply scenario adjustments.
        forecast_df = adjust_forecast(forecast_df, demand_shock, seasonality_adjustment, external_shock, category_scenarios)

        # Inverse differencing if needed.
        if is_diff and last_historical_value is not None:
            forecast_df = inverse_difference(forecast_df, last_historical_value)

        match_len = min(len(test["y"]), len(forecast_df))
        rmse = mean_squared_error(test["y"].iloc[:match_len], forecast_df["yhat"].iloc[:match_len], squared=False)
        mape = mean_absolute_percentage_error(test["y"].iloc[:match_len], forecast_df["yhat"].iloc[:match_len])

        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast_df}

    except Exception as e:
        st.warning(f"ARIMA Model failed: {e}")

    return "ARIMA", result

import calendar
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def train_xgb_model(train, test, forecast_period, last_historical_value, is_diff,
                    demand_shock, seasonality_adjustment, external_shock, category_scenarios=None):
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
        feature_cols = [c for c in data_xgb.columns if c not in ["y","ds"] and np.issubdtype(data_xgb[c].dtype, np.number)]

        model = XGBRegressor(
            n_estimators=50,
            max_depth=min(5, max(2, len(train)//10)),
            learning_rate=0.1 if len(train)>50 else 0.2,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
        model.fit(data_xgb[feature_cols], data_xgb["y"])

        # Generate future forecasts
        last_row = data_xgb.iloc[-1].copy()
        last_date = train["ds"].iloc[-1]
        preds = []
        for i in range(forecast_period):
            future_date = last_date + pd.DateOffset(months=i+1)
            future = {col: last_row[col] for col in feature_cols}
            future.update({
                "month": future_date.month,
                "quarter": future_date.quarter,
                "year": future_date.year,
                "sin_month": np.sin(2*np.pi*future_date.month/12),
                "cos_month": np.cos(2*np.pi*future_date.month/12)
            })
            pred = model.predict(pd.DataFrame([future]))[0]
            preds.append(pred)
            for lag in range(max_lag,1,-1):
                last_row[f"lag_{lag}"] = last_row[f"lag_{lag-1}"]
            last_row["lag_1"] = pred

        forecast_df = pd.DataFrame({
            "ds": pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_period, freq="MS"),
            "yhat": preds
        })
        forecast_df = adjust_forecast(forecast_df, demand_shock, seasonality_adjustment, external_shock, category_scenarios)

        match_len = min(len(test), len(forecast_df))
        rmse = mean_squared_error(test["y"].iloc[:match_len], forecast_df["yhat"].iloc[:match_len], squared=False)
        mape = mean_absolute_percentage_error(test["y"].iloc[:match_len], forecast_df["yhat"].iloc[:match_len])

        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast_df}

    except Exception as e:
        st.warning(f"XGBoost Model failed: {e}")

    return "XGBoost", result

def train_automl_model(train, test, forecast_period, last_historical_value, is_diff, 
                       demand_shock, seasonality_adjustment, external_shock, category_scenarios=None, time_budget=None):
    """
    Train an AutoML model using FLAML for time series forecasting.

    Args:
        train (pd.DataFrame): Training data with columns "ds" (date) and "y" (target).
        test (pd.DataFrame): Test data for evaluation.
        forecast_period (int): Number of periods to forecast.
        last_historical_value (float): Last observed value before differencing (if applied).
        is_diff (bool): Indicates whether differencing was applied.
        demand_shock (float): Global demand shock adjustment percentage.
        seasonality_adjustment (float): Seasonality adjustment percentage.
        external_shock (bool): Whether to apply an external shock adjustment.
        category_scenarios (dict, optional): Category-specific forecast adjustments.
        time_budget (int or None): Time budget in seconds for AutoML training. If None, it is dynamically calculated.

    Returns:
        tuple: Model name ("AutoML") and a dictionary containing RMSE, MAPE, and forecast DataFrame.
    """
    result = {}
    try:
        # Feature Engineering: work on a copy of the training data.
        data_automl = train.copy()

        # If category adjustments are used, aggregate the data by date (summing up "y")
        if category_scenarios:
            data_automl = data_automl.groupby("ds", as_index=False).agg({"y": "sum"})

        # Use the aggregated dataset's length for subsequent calculations.
        n = len(data_automl)

        # Determine maximum lag based on aggregated dataset size
        max_lag = min(24, n - 1)
        if n <= 6:
            max_lag = min(3, n - 1)
        elif n <= 12:
            max_lag = min(6, n - 1)
        elif n <= 24:
            max_lag = min(12, n - 1)

        # Add lag features
        for lag in range(1, max_lag + 1):
            data_automl[f"lag_{lag}"] = data_automl["y"].shift(lag)

        # Add rolling statistics
        for window in [3, 6, 12]:
            data_automl[f"rolling_mean_{window}"] = data_automl["y"].rolling(window=window, min_periods=1).mean()
            data_automl[f"rolling_std_{window}"] = data_automl["y"].rolling(window=window, min_periods=1).std()

        # Add year-over-year growth (if sufficient data)
        if n > 12:
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

        # Drop rows with missing values so that X and y align
        data_automl.dropna(inplace=True)

        # Prepare features and target
        feature_cols = [col for col in data_automl.columns if col not in ["y", "ds", "y_log"]]
        y_train = data_automl["y_log"] if apply_log else data_automl["y"]
        X_train = data_automl[feature_cols]

        # Dynamic time budget calculation using the aggregated data length
        if time_budget is None:
            time_budget = min(600, max(60, n * 0.1 + len(feature_cols) * 2))  # 60s to 600s
            st.info(f"Dynamic time budget set to {time_budget} seconds based on dataset size and complexity.")

        # Train AutoML model using FLAML
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
            "ds": pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1),
                                 periods=forecast_period, freq="MS"),
            "yhat": automl_forecast,
            "yhat_lower": automl_forecast * 0.9,
            "yhat_upper": automl_forecast * 1.1
        })

        # Inverse differencing if applicable
        if isinstance(last_historical_value, (int, float)):
            forecast_df["yhat"] = inverse_difference(forecast_df["yhat"], last_historical_value)
            forecast_df["yhat_lower"] = inverse_difference(forecast_df["yhat_lower"], last_historical_value)
            forecast_df["yhat_upper"] = inverse_difference(forecast_df["yhat_upper"], last_historical_value)

        # Apply forecast adjustments
        forecast_df = adjust_forecast(forecast_df, demand_shock, seasonality_adjustment, external_shock, category_scenarios)

        # Evaluate model performance: use the minimum length from test and forecast
        match_len = min(len(test["y"]), len(forecast_df))
        rmse = np.sqrt(mean_squared_error(test["y"].values[:match_len], forecast_df["yhat"].iloc[:match_len]))
        mape = mean_absolute_percentage_error(test["y"].values[:match_len], forecast_df["yhat"].iloc[:match_len])
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
    # Set subscription level: "free" for basic features, "premium" for full access
    user_id = "user123"
    subscription_level = "premium"  # Change to "premium" to enable advanced features

    if subscription_level != "premium":
        st.info("You are using the Free version. Advanced features such as category adjustments, extended forecast horizons, hyperparameter tuning, and forecast downloads are disabled.")

    if st.button("🔄 Reset App"):
        st.session_state.clear()
        st.experimental_rerun()

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
                """, unsafe_allow_html=True)

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
                    if subscription_level != "premium":
                        st.info("Category adjustments are available only for premium users.")
                        category_columns = None
                    else:
                        category_columns = st.multiselect(
                            "🏷️ Select Category Columns (Optional):",
                            options=[col for col in data.columns if col not in [date_column, sales_column]],
                            key="category_cols"
                        )

            if date_column != "-- Select Column --":
                data[date_column] = pd.to_datetime(data[date_column], errors="coerce")

            if subscription_level != "premium":
                time_budget = 60
                forecast_period = 3
            else:
                st.markdown("### ⏱️ AutoML Time Budget")
                time_budget = st.slider(
                    "Set the time budget for AutoML training (in seconds):",
                    min_value=60, max_value=1200, value=300, step=60,
                    help="Increase the time budget for larger datasets or more complex models."
                )
                forecast_period = 24

            if date_column != "-- Select Column --" and sales_column != "-- Select Column --":
                last_date_in_data = data[date_column].max()
                min_future_date = (last_date_in_data + pd.DateOffset(days=1)).date()

                if subscription_level != "premium":
                    st.sidebar.info("Scenario planning is available only for premium users.")
                    demand_shock = 0
                    seasonality_adjustment = 0
                    external_shock = False
                    category_scenarios = {}
                else:
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
                    category_scenarios = {}
                    if category_columns:
                        st.sidebar.markdown("### 🎯 Scenario Planning by Category (Dynamic)")
                        for col in category_columns:
                            st.sidebar.markdown(f"#### Adjustments for '{col}'")
                            unique_cats = sorted(data[col].dropna().unique())
                            selected_cats = st.sidebar.multiselect(
                                f"Pick categories in '{col}' to adjust:",
                                options=unique_cats,
                                help=f"Select one or more categories from '{col}' that you want to adjust."
                            )
                            category_scenarios[col] = {}
                            for cat in selected_cats:
                                with st.sidebar.expander(f"Adjust '{cat}' in '{col}'"):
                                    cat_adjust = st.slider(
                                        f"Percentage change for '{cat}'",
                                        min_value=-50, max_value=50, value=0, step=5,
                                        help=f"Adjust sales for category '{cat}' within column '{col}'"
                                    )
                                    cat_start = st.date_input(
                                        f"Start date for '{cat}'",
                                        value=min_future_date,
                                        min_value=min_future_date
                                    )
                                    default_end = (last_date_in_data + pd.DateOffset(months=1)).date()
                                    cat_end = st.date_input(
                                        f"End date for '{cat}'",
                                        value=default_end if default_end > min_future_date else min_future_date,
                                        min_value=cat_start
                                    )
                                    category_scenarios[col][cat] = {
                                        "adjustment": cat_adjust,
                                        "start_date": cat_start,
                                        "end_date": cat_end
                                    }
                    else:
                        st.sidebar.warning("Please select the Date and Sales columns to enable scenario planning.")
                        demand_shock = 0
                        seasonality_adjustment = 0
                        external_shock = False
                        category_scenarios = {}

            if date_column != "-- Select Column --" and sales_column != "-- Select Column --":
                start_forecast = st.button("✅ Start Forecast", key="start_btn",
                                           help="Click to generate your AI-powered forecast")
            else:
                start_forecast = st.button("⏳ Select Columns First", disabled=True, key="start_disabled")

            if subscription_level == "premium":
                total_steps = 12
            else:
                total_steps = 6

            overall_status = st.empty()
            prophet_status = st.empty()
            arima_status = st.empty()
            xgb_status = st.empty()
            automl_status = st.empty()
            progress_bar = st.progress(0)
            step_message = st.empty()

            if start_forecast:
                # STEP 1: Preprocess Data
                step = 1
                step_message.text(f"Step {step} of {total_steps}: Preprocessing data...")
                with st.spinner("🔍 Preprocessing data..."):
                    processed_data, last_historical_value, y_original, is_diff = preprocess_data(
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

                # STEP 2: Review Processed Data
                step += 1
                step_message.text(f"Step {step} of {total_steps}: Reviewing processed data...")
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <h2 style="color: #2B3A42;">📅 Preprocessed Monthly Data</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with st.expander("📊 View Processed Data"):
                    st.dataframe(processed_data.style.set_properties(**{"text-align": "center"}), width=1400, height=450)
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(0.5)

                # STEP 3: Split Data into Training and Test Sets
                step += 1
                step_message.text(f"Step {step} of {total_steps}: Splitting data into training and test sets...")
                testing_period = int(len(processed_data) * 0.2)
                train = processed_data.iloc[:-testing_period]
                test = processed_data.iloc[-testing_period:]
                progress_bar.progress(int((step / total_steps) * 100))
                time.sleep(1)

                if subscription_level == "premium":
                    # PREMIUM PIPELINE
                    # STEP 4: Tune Prophet Model
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Tuning Prophet model...")
                    with st.spinner("🚀 Tuning Prophet model..."):
                        best_params, best_rmse = find_best_prophet_params(train)
                        time.sleep(1)
                    if best_params is None:
                        st.error("No valid Prophet parameters were found.")
                        return
                    st.success(f"✅ Best Prophet Params: {best_params}")
                    overall_status.write(f"📉 Best RMSE (CV): {best_rmse:.2f}")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # STEP 5: Train Prophet Model
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Training Prophet model...")
                    with st.spinner("🚀 Training Prophet model..."):
                        prophet_model_name, prophet_res = train_prophet_model(
                            train, test, forecast_period, best_params,
                            last_historical_value, is_diff,
                            demand_shock, seasonality_adjustment, external_shock, category_scenarios
                        )
                        time.sleep(1)
                    if is_diff and prophet_res.get("Forecast") is not None:
                        prophet_res["Forecast"] = inverse_difference(prophet_res["Forecast"], last_historical_value)
                    st.success("✅ Prophet Model Training Complete!")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # # STEP 6: Train ARIMA Model
                    # step += 1
                    # step_message.text(f"Step {step} of {total_steps}: Training ARIMA model...")
                    # with st.spinner("🚀 Training ARIMA model..."):
                    #     arima_model_name, arima_res = train_arima_model(
                    #         train, test, forecast_period,
                    #         last_historical_value, is_diff,
                    #         demand_shock, seasonality_adjustment, external_shock, category_scenarios
                    #     )
                    #     time.sleep(1)
                    # if is_diff and arima_res.get("Forecast") is not None:
                    #     arima_res["Forecast"] = inverse_difference(arima_res["Forecast"], last_historical_value)
                    # st.success("✅ ARIMA Model Training Complete!")
                    # progress_bar.progress(int((step / total_steps) * 100))
                    # time.sleep(1)

                    # STEP 7: Train XGBoost Model
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Training XGBoost model...")
                    with st.spinner("🚀 Training XGBoost model..."):
                        xgb_model_name, xgb_res = train_xgb_model(
                            train, test, forecast_period,
                            last_historical_value, is_diff,
                            demand_shock, seasonality_adjustment, external_shock, category_scenarios
                        )
                        time.sleep(1)
                    if is_diff and xgb_res.get("Forecast") is not None:
                        xgb_res["Forecast"] = inverse_difference(xgb_res["Forecast"], last_historical_value)
                    st.success("✅ XGBoost Model Training Complete!")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # STEP 9: Train AutoML Model
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Training AutoML model...")
                    with st.spinner("🚀 Training AutoML Model..."):
                        automl_model_name, automl_res = train_automl_model(
                            train, test, forecast_period,
                            last_historical_value, is_diff,
                            demand_shock, seasonality_adjustment, external_shock,
                            category_scenarios, time_budget
                        )
                        time.sleep(1)
                    st.success("✅ AutoML Model Training Complete!")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # STEP 9: Compile Forecast Results (Premium)
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Compiling forecast results...")
                    st.success("🎉 Forecasting process completed!")
                    time.sleep(1)
                    results = {
                        prophet_model_name: prophet_res,
                        # arima_model_name: arima_res,
                        xgb_model_name: xgb_res,
                        automl_model_name: automl_res
                    }
                    valid_results = {model: res for model, res in results.items() if res.get("Forecast") is not None}
                    if not valid_results:
                        st.error("No valid model forecasts produced.")
                        return
                    st.session_state.model_results = valid_results
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # STEP 10: Display Model Performance Comparison (Premium)
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Displaying model performance comparison...")
                    comparison_data = []
                    for model, res in st.session_state.model_results.items():
                        forecast_df = res["Forecast"]
                        match_len = min(len(test["y"]), len(forecast_df))
                        actual = test["y"].iloc[:match_len].values
                        pred = forecast_df["yhat"].iloc[:match_len].values
                        corr = shape_score(actual, pred)
                        res["Shape (corr)"] = corr
                        comparison_data.append({
                            "Model": model,
                            "RMSE": res["RMSE"],
                            "MAPE": res["MAPE"],
                            "Shape (corr)": res["Shape (corr)"]
                        })
                    if comparison_data:
                        max_rmse = max(res["RMSE"] for res in st.session_state.model_results.values())
                        for model, res in st.session_state.model_results.items():
                            res["Combined Score"] = combined_score(res["RMSE"], res["Shape (corr)"], max_rmse, 0.5, 1.0)
                        for item in comparison_data:
                            item["Combined Score"] = combined_score(item["RMSE"], item["Shape (corr)"], max_rmse, 0.5, 1.0)
                        comparison_df = pd.DataFrame(comparison_data).sort_values(by="Combined Score")
                        st.dataframe(comparison_df.style.highlight_min(subset=["Combined Score"], color="lightgreen"))
                        best_model = comparison_df.iloc[0]["Model"]
                        st.success(f"✨ **AI-Selected Best Model (Combined):** {best_model}")
                    else:
                        st.warning("No model results found. Please train the models first.")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # 🔮 AI-Powered Future Insights + Category Summary + High-Risk Detection
                    try:
                        # Retrieve the best model's forecast DataFrame
                        forecast_data = st.session_state.model_results[best_model]["Forecast"]

                        # Compute AI-powered insights
                        highest_point = forecast_data.loc[forecast_data["yhat"].idxmax()]
                        lowest_point = forecast_data.loc[forecast_data["yhat"].idxmin()]
                        projected_growth = ((forecast_data["yhat"].iloc[-1] - test["y"].iloc[-1]) / test["y"].iloc[-1]) * 100
                        trend = "📈 **Growth Expected**" if projected_growth > 0 else "📉 **Potential Decline**"

                        insights_text = f"""
                    - **Projected Sales Growth:** {abs(projected_growth):.2f}% {trend}
                    - **Peak Sales Expected:** ${highest_point['yhat']:.2f} on {highest_point['ds'].strftime('%Y-%m-%d')}
                    - **Lowest Predicted Sales:** ${lowest_point['yhat']:.2f} on {lowest_point['ds'].strftime('%Y-%m-%d')}
                    - **Optimal Decision Window:** Plan around peak sales in {highest_point['ds'].strftime('%B %Y')}
                    - **Risk Zones Identified:** Check months marked as 🔥 'High-Risk' below
                    - **Volatility Analysis:** Forecast suggests a {'stable' if abs(projected_growth) < 5 else 'fluctuating'} trend
                        """

                        with st.expander("🔮 AI-Powered Future Insights", expanded=True):
                            st.markdown(insights_text)

                        # Build a summary of category adjustments if any were applied.
                        if category_scenarios:
                            cat_adj_summary = "### Category Adjustments Summary\n"
                            for col, adjustments in category_scenarios.items():
                                cat_adj_summary += f"- **{col}**:\n"
                                for cat, details in adjustments.items():
                                    cat_adj_summary += (
                                        f"  - **{cat}**: {details['adjustment']}% adjustment "
                                        f"from {details['start_date']} to {details['end_date']}\n"
                                    )
                            st.markdown(cat_adj_summary)

                        # 🔥 Detect High-Risk Periods in Forecast
                        if forecast_data is not None:
                            try:
                                forecast_data["volatility"] = forecast_data["yhat"].rolling(3).std()
                                forecast_data["risk"] = "✅ Stable"

                                # Define thresholds at 75th and 90th percentile
                                p75 = forecast_data["volatility"].quantile(0.75)
                                p90 = forecast_data["volatility"].quantile(0.90)

                                forecast_data.loc[forecast_data["volatility"] > p75, "risk"] = "⚠️ High Volatility"
                                forecast_data.loc[forecast_data["volatility"] > p90, "risk"] = "❌ Major Decline"

                                st.markdown("### 🚨 High-Risk Sales Periods Identified")
                                st.dataframe(
                                    forecast_data[["ds", "yhat", "volatility", "risk"]]
                                    .style.applymap(
                                        lambda x: (
                                            "background-color: #FFDDC1" if x == "❌ Major Decline" else
                                            "background-color: #FFEEAA" if x == "⚠️ High Volatility" else
                                            "background-color: #C6ECAE"
                                        ),
                                        subset=["risk"]
                                    )
                                )
                            except Exception as e:
                                st.error(f"❌ Error detecting high-risk periods: {e}")

                    except Exception as e:
                        st.error(f"❌ Error analyzing forecast data: {e}")

                    # STEP 11: Finalize Forecast Visualization (Premium)
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Finalizing forecast visualization...")
                    st.markdown("### 🔍 Forecast Comparison Across Models")
                    model_colors = {
                        "Prophet": "blue",
                        # "ARIMA": "green",
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
                        csv = st.session_state.model_results[best_model]["Forecast"].to_csv(index=False)
                        st.download_button(
                            label="📩 Download Best Model Forecast (CSV)",
                            data=csv,
                            file_name="forecast.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"❌ Error generating download file: {e}")
                else:
                    # FREE USERS PIPELINE (only AutoML)
                    # STEP 4 (Free): Train AutoML Model
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Training AutoML model (Limited)...")
                    with st.spinner("🚀 Training AutoML Model..."):
                        automl_model_name, automl_res = train_automl_model(
                            train, test, forecast_period,
                            last_historical_value, is_diff,
                            time_budget, demand_shock, seasonality_adjustment, external_shock, category_scenarios
                        )
                        time.sleep(1)
                    if is_diff and automl_res.get("Forecast") is not None:
                        automl_res["Forecast"] = inverse_difference(automl_res["Forecast"], last_historical_value)
                    automl_status.success("✅ AutoML Model Training Complete!")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)
                    
                    # STEP 5 (Free): Compile Forecast Results
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Compiling forecast results...")
                    st.success("🎉 Forecasting process completed!")
                    time.sleep(1)
                    results = {"AutoML": automl_res}
                    st.session_state.model_results = results
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)
                    
                    # STEP 6 (Free): Finalize Forecast Visualization
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Finalizing forecast visualization...")
                    st.markdown("### 🔍 Forecast Visualization")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_original["ds"],
                        y=y_original["y_original"],
                        mode="lines",
                        name="Historical Data",
                        line=dict(color="black", width=2)
                    ))
                    forecast_df = automl_res["Forecast"]
                    fig.add_trace(go.Scatter(
                        x=forecast_df["ds"],
                        y=forecast_df["yhat"],
                        mode="lines",
                        name="AutoML Forecast",
                        line=dict(width=2, color="purple")
                    ))
                    fig.update_layout(
                        title="📊 Sales Forecast",
                        xaxis_title="Date",
                        yaxis_title="Sales",
                        template="plotly_white",
                        xaxis_tickformat="%Y-%m"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    progress_bar.progress(100)
                    step_message.text("All steps completed!")
                    
                    st.info("Upgrade to Premium to unlock advanced features like multi-model comparison and forecast download.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()