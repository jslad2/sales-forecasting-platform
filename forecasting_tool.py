# ==== Python & ML Libraries ====
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
from datetime import datetime
import time
import os
import uuid
import json
import calendar
from scipy.stats import pearsonr
import concurrent.futures
from tqdm import tqdm
import catboost

# ==== Supabase ====
from supabase import create_client
from dotenv import load_dotenv
from supabase_utils import upload_forecast


# ==== Streamlit Config ====
st.set_page_config(
    layout="wide",
    page_title="Time Series Forecasting",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ==== Load Environment Variables FIRST ====
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ==== Query Params from Dashboard ====
query_params = st.experimental_get_query_params()
user_id = query_params.get("user_id", ["guest"])[0]
subscription_level = query_params.get("subscription_level", ["free"])[0]

# Optional Welcome Message
st.markdown(f"### Welcome, {user_id}!")
st.markdown(f"Your Subscription Level: **{subscription_level.capitalize()}**")

# Free Tier Notice
if subscription_level != "premium":
    st.info("You are using the Free version. Advanced features such as category adjustments, extended forecast horizons, hyperparameter tuning, and forecast downloads are disabled.")


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
        param_grid = {
            "changepoint_prior_scale": [0.01, 0.1],
            "seasonality_mode": ["additive"],
            "seasonality_prior_scale": [10.0]  # for smaller datasets, maybe less sensitivity
        }
    else:
        param_grid = {
            "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.2, 0.3],
            "seasonality_mode": ["additive", "multiplicative"],
            "seasonality_prior_scale": [1.0, 5.0, 10.0]  # wider range for larger datasets
        }
        
    # Adjust forecasting horizons based on dataset size
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
    
    import concurrent.futures
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
                        is_diff, demand_shock, seasonality_adjustment, external_shock,
                        category_scenarios=None, holidays=None, growth_type='linear'):
    """
    Trains a Prophet model with improvements:
      - Outlier clipping on y values.
      - Option to choose between 'linear' and 'logistic' growth.
      - Optional holiday effects.
      - Enhanced seasonalities.
    """
    result = {}
    try:
        # If category adjustments are provided, aggregate training data.
        if category_scenarios:
            train = train.groupby("ds", as_index=False).agg({"y": "sum"})

        # Outlier handling: clip extreme values using the 5th and 95th percentiles.
        lower_bound, upper_bound = train["y"].quantile([0.05, 0.95])
        train["y"] = train["y"].clip(lower=lower_bound, upper=upper_bound)

        # Set growth parameters based on growth_type.
        if growth_type == "logistic":
            # Use a slightly tighter cap than before.
            train["cap"] = 1.1 * train["y"].max()
            train["floor"] = max(train["y"].min() * 0.9, 1)  # ensure minimal positive floor
            model = Prophet(
                growth="logistic",
                seasonality_mode=best_params["seasonality_mode"],
                changepoint_prior_scale=best_params["changepoint_prior_scale"],
                holidays=holidays,
                yearly_seasonality=(len(train) >= 365),
                weekly_seasonality=(len(train) >= 30),
                daily_seasonality=False
            )
        else:
            model = Prophet(
                growth="linear",
                seasonality_mode=best_params["seasonality_mode"],
                changepoint_prior_scale=best_params["changepoint_prior_scale"],
                holidays=holidays,
                yearly_seasonality=(len(train) >= 365),
                weekly_seasonality=(len(train) >= 30),
                daily_seasonality=False
            )

        # Add custom seasonalities robustly.
        model = detect_and_add_seasonalities(model, train)

        # Fit the model.
        model.fit(train)

        # Create future dataframe.
        future = model.make_future_dataframe(periods=forecast_period, freq="MS", include_history=False)
        if growth_type == "logistic":
            future["cap"] = train["cap"].max()
            future["floor"] = train["floor"].min()

        forecast = model.predict(future)
        forecast = forecast[forecast["ds"] > train["ds"].max()]

        # Clamp forecasts to be nonnegative.
        forecast["yhat"] = forecast["yhat"].clip(lower=0)

        # Apply any global or category-specific adjustments.
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
        st.warning(f"Updated Prophet Model failed: {e}")

    return "Prophet", result

def train_arima_model(train, test, forecast_period, last_historical_value, is_diff,
                      demand_shock, seasonality_adjustment, external_shock, category_scenarios=None):
    """
    Trains an ARIMA model with improvements:
      - Automatic inference of seasonal frequency.
      - Robust seasonal decomposition and auto_arima configuration.
    """
    result = {}
    try:
        if category_scenarios:
            train = train.groupby("ds", as_index=False).agg({"y": "sum"})

        # Infer the frequency to set the seasonal period.
        inferred_freq = pd.infer_freq(train["ds"])
        if inferred_freq is None:
            seasonal_period = 12
        elif inferred_freq in ['MS', 'M']:
            seasonal_period = 12
        elif inferred_freq in ['D']:
            seasonal_period = 7
        else:
            seasonal_period = 12

        # Use seasonal decomposition to flag seasonality.
        try:
            decomposition = seasonal_decompose(train["y"], model="additive", period=seasonal_period)
            seasonal_present = np.any(np.abs(decomposition.seasonal) > 0.01)
            acf_values = acf(train["y"], nlags=seasonal_period, fft=False)
            seasonal_confirmed = any(np.abs(acf_values[1:]) > 0.2)
            seasonal = seasonal_present and seasonal_confirmed
        except Exception as e:
            st.warning(f"Seasonality analysis failed: {e}")
            seasonal = False

        # Fit auto_arima with tighter approximation settings.
        model = auto_arima(
            train["y"],
            seasonal=seasonal,
            m=seasonal_period if seasonal else 1,
            d=None,
            D=1 if seasonal else 0,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True,
            approximation=False
        )

        preds = model.predict(n_periods=forecast_period)
        forecast_dates = pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1),
                                       periods=len(preds), freq="MS")
        forecast_df = pd.DataFrame({"ds": forecast_dates, "yhat": preds})

        # Apply adjustments.
        forecast_df = adjust_forecast(forecast_df, demand_shock, seasonality_adjustment, external_shock, category_scenarios)

        # Inverse differencing if needed.
        if is_diff and last_historical_value is not None:
            forecast_df = inverse_difference(forecast_df, last_historical_value)

        match_len = min(len(test["y"]), len(forecast_df))
        rmse = mean_squared_error(test["y"].iloc[:match_len], forecast_df["yhat"].iloc[:match_len], squared=False)
        mape = mean_absolute_percentage_error(test["y"].iloc[:match_len], forecast_df["yhat"].iloc[:match_len])
        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast_df}

    except Exception as e:
        st.warning(f"Updated ARIMA Model failed: {e}")

    return "ARIMA", result

def train_xgb_model(train, test, forecast_period, last_historical_value, is_diff,
                    demand_shock, seasonality_adjustment, external_shock, category_scenarios=None):
    """
    Trains an XGBoost model with improvements:
      - Outlier clipping to smooth extreme values.
      - Expanded lag features and seasonality indicators.
      - Updated hyperparameters for robustness.
    """
    result = {}
    try:
        if category_scenarios:
            train = train.groupby("ds", as_index=False).agg({"y": "sum"})

        # Outlier handling: clip extreme y values.
        lower_bound, upper_bound = train["y"].quantile([0.05, 0.95])
        train["y"] = train["y"].clip(lower=lower_bound, upper=upper_bound)

        # Determine peak month robustly.
        monthly_avg = train.groupby(train["ds"].dt.month)["y"].mean()
        peak_month = monthly_avg.idxmax()
        peak_value = monthly_avg.max()

        # Feature engineering.
        data_xgb = train.copy()
        data_xgb["month"] = data_xgb["ds"].dt.month
        data_xgb["year"] = data_xgb["ds"].dt.year
        data_xgb["month_year_interaction"] = data_xgb["month"] * (data_xgb["year"] - data_xgb["year"].min())
        data_xgb["days_from_peak"] = (data_xgb["ds"].dt.month - peak_month) % 12
        data_xgb["is_peak_month"] = (data_xgb["ds"].dt.month == peak_month).astype(int)
        phase_shift = peak_month - 1
        data_xgb["sin_month"] = np.sin(2 * np.pi * (data_xgb["month"] - phase_shift) / 12)
        data_xgb["cos_month"] = np.cos(2 * np.pi * (data_xgb["month"] - phase_shift) / 12)
        data_xgb["sin2_month"] = np.sin(4 * np.pi * (data_xgb["month"] - phase_shift) / 12)
        data_xgb["cos2_month"] = np.cos(4 * np.pi * (data_xgb["month"] - phase_shift) / 12)

        # Create multiple lag features.
        max_lag = min(24, len(train) - 1)
        for lag in [1, 2, 3, 6, 12, 24]:
            if lag <= max_lag:
                data_xgb[f"lag_{lag}"] = data_xgb["y"].shift(lag)
        data_xgb.dropna(inplace=True)

        feature_order = [
            'month', 'year', 'month_year_interaction', 'is_peak_month', 'days_from_peak',
            'sin_month', 'cos_month', 'sin2_month', 'cos2_month'
        ] + [f"lag_{lag}" for lag in [1, 2, 3, 6, 12, 24] if f"lag_{lag}" in data_xgb.columns]
        feature_cols = feature_order

        # Updated hyperparameters.
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        model.fit(data_xgb[feature_cols], data_xgb["y"])

        # Recalculate a dynamic boost factor based on recent peak performance.
        recent_years = sorted(train["ds"].dt.year.unique())[-3:]
        recent_data = train[train["ds"].dt.year.isin(recent_years)]
        recent_monthly_avg = recent_data.groupby(recent_data["ds"].dt.month)["y"].mean()
        if recent_monthly_avg.mean() != 0:
            dynamic_boost = recent_monthly_avg.get(peak_month, peak_value) / recent_monthly_avg.mean()
        else:
            dynamic_boost = 1.0
        dynamic_boost = np.clip(dynamic_boost, 1.0, 1.2)

        # Year-over-year growth calculation.
        yearly_peaks = {}
        for year in sorted(train["ds"].dt.year.unique()):
            year_data = train[(train["ds"].dt.year == year) & (train["ds"].dt.month == peak_month)]
            if not year_data.empty:
                yearly_peaks[year] = year_data["y"].mean()
        growth_factors = []
        unique_years = sorted(yearly_peaks.keys())
        for i in range(len(unique_years) - 1):
            if yearly_peaks[unique_years[i]] > 0:
                growth_factors.append(yearly_peaks[unique_years[i+1]] / yearly_peaks[unique_years[i]])
        yoy_growth = np.mean(growth_factors) if growth_factors else 1.0
        yoy_growth = max(yoy_growth, 1.0)
        last_training_year = unique_years[-1] if unique_years else data_xgb["year"].max()

        # Forecast generation.
        last_row = data_xgb[feature_cols].iloc[-1].copy()
        last_date = data_xgb["ds"].iloc[-1]
        preds = []
        for i in range(forecast_period):
            future_date = last_date + pd.DateOffset(months=i+1)
            is_peak = int(future_date.month == peak_month)
            future_year = future_date.year
            years_ahead = future_year - last_training_year
            compounded_boost = dynamic_boost * (yoy_growth ** years_ahead)
            compounded_boost = max(compounded_boost, 1.0)
            features = {col: last_row[col] for col in feature_cols}
            features.update({
                'month': future_date.month,
                'year': future_year,
                'month_year_interaction': future_date.month * (future_year - data_xgb["year"].min()),
                'is_peak_month': is_peak,
                'days_from_peak': (future_date.month - peak_month) % 12,
                'sin_month': np.sin(2 * np.pi * (future_date.month - phase_shift) / 12),
                'cos_month': np.cos(2 * np.pi * (future_date.month - phase_shift) / 12),
                'sin2_month': np.sin(4 * np.pi * (future_date.month - phase_shift) / 12),
                'cos2_month': np.cos(4 * np.pi * (future_date.month - phase_shift) / 12)
            })
            base_pred = model.predict(pd.DataFrame([features]))[0]
            pred = base_pred * compounded_boost if is_peak else base_pred
            preds.append(pred)
            # Update lag features for iterative forecasting.
            for lag in [24, 12, 6, 3, 2, 1]:
                if f'lag_{lag}' in feature_cols:
                    if lag == 1:
                        last_row['lag_1'] = pred
                    else:
                        last_row[f'lag_{lag}'] = last_row.get(f'lag_{lag-1}', pred)

        forecast_df = pd.DataFrame({
            "ds": pd.date_range(start=last_date + pd.DateOffset(months=1),
                                periods=forecast_period, freq="MS"),
            "yhat": preds
        })

        if is_diff and last_historical_value is not None:
            forecast_df["yhat"] = last_historical_value + forecast_df["yhat"].cumsum()

        forecast_df = adjust_forecast(forecast_df, demand_shock, seasonality_adjustment, external_shock, category_scenarios)

        match_len = min(len(test), len(forecast_df))
        rmse = mean_squared_error(test["y"].iloc[:match_len], forecast_df["yhat"].iloc[:match_len], squared=False)
        mape = mean_absolute_percentage_error(test["y"].iloc[:match_len], forecast_df["yhat"].iloc[:match_len])
        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast_df}

    except Exception as e:
        st.warning(f"Updated XGBoost Model failed: {str(e)}")
        result = {"error": str(e)}

    return "XGBoost", result

def dynamic_rmse_metric(X_val, y_val, estimator, labels, 
                          X_train, y_train, weight_val=None, weight_train=None, 
                          config=None, groups_val=None, groups_train=None):
    """
    Custom dynamic RMSE that penalizes errors more strongly near peak values.
    
    This function computes the RMSE between y_val and the predictions from the estimator
    on X_val, with higher weights applied to samples above 90% of the maximum true value.
    If the sizes of y_val and the predictions differ (e.g. due to cross-validation),
    the larger array is aggregated (averaged) to match the size of the smaller.
    
    Returns:
        tuple: (weighted_rmse, {"dynamic_rmse": weighted_rmse})
               where weighted_rmse is a float that should be minimized.
    """
    # Get predictions from the estimator on the validation set.
    y_pred = estimator.predict(X_val)
    
    # Convert true values and predictions to flattened numpy arrays.
    y_true = np.asarray(y_val).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    n_true = len(y_true)
    n_pred = len(y_pred)
    
    # If lengths differ, aggregate the larger one.
    if n_true != n_pred:
        if n_true > n_pred:
            factor = n_true // n_pred
            if factor * n_pred == n_true:
                y_true = y_true.reshape(n_pred, factor).mean(axis=1)
            else:
                y_true = np.array([np.mean(block) for block in np.array_split(y_true, n_pred)])
        elif n_pred > n_true:
            factor = n_pred // n_true
            if factor * n_true == n_pred:
                y_pred = y_pred.reshape(n_true, factor).mean(axis=1)
            else:
                y_pred = np.array([np.mean(block) for block in np.array_split(y_pred, n_true)])
    
    # Sanity check: ensure that after aggregation, the shapes match.
    if len(y_true) != len(y_pred):
        raise ValueError(f"After aggregation, shapes still don't match: y_true {y_true.shape}, y_pred {y_pred.shape}")
    
    # Compute dynamic weighting: set a threshold at 90% of the maximum true value.
    peak_threshold = 0.9 * np.max(y_true)
    # Assign a higher weight (2.0) to values at or above this threshold.
    weights = np.where(y_true >= peak_threshold, 2.0, 1.0)
    
    # Compute weighted squared errors and the resulting RMSE.
    weighted_squared_errors = weights * (y_pred - y_true)**2
    weighted_rmse = np.sqrt(np.mean(weighted_squared_errors))
    
    # Return a tuple: the score (to be minimized) and a dictionary for logging.
    return weighted_rmse, {"dynamic_rmse": weighted_rmse}

def train_automl_model(train, test, forecast_period, last_historical_value, is_diff, 
                       demand_shock, seasonality_adjustment, external_shock, category_scenarios=None, time_budget=None):
    """
    Trains an AutoML model using FLAML with improved features including a dynamic, peak-sensitive metric.
    """
    result = {}
    try:
        data_automl = train.copy()
        if category_scenarios:
            data_automl = data_automl.groupby("ds", as_index=False).agg({"y": "sum"})

        n = len(data_automl)
        max_lag = min(24, n - 1)
        if n <= 6:
            max_lag = min(3, n - 1)
        elif n <= 12:
            max_lag = min(6, n - 1)
        elif n <= 24:
            max_lag = min(12, n - 1)

        # Create lag features, rolling statistics, and any additional features
        for lag in range(1, max_lag + 1):
            data_automl[f"lag_{lag}"] = data_automl["y"].shift(lag)
        for window in [3, 6, 12]:
            data_automl[f"rolling_mean_{window}"] = data_automl["y"].rolling(window=window, min_periods=1).mean()
            data_automl[f"rolling_std_{window}"] = data_automl["y"].rolling(window=window, min_periods=1).std()
            # Rolling maximum feature for peak detection.
            data_automl[f"rolling_max_{window}"] = data_automl["y"].rolling(window=window, min_periods=1).max()

        # Create a peak indicator feature.
        historical_peak = data_automl["y"].max()
        data_automl["is_peak_month"] = (data_automl["y"] >= 0.9 * historical_peak).astype(int)
        
        # Add growth and transformation features.
        if n > 12:
            data_automl["yoy_growth"] = (data_automl["y"] / data_automl["y"].shift(12)) - 1
        else:
            data_automl["yoy_growth"] = 0
        data_automl["y_diff"] = data_automl["y"].diff().fillna(0)
        data_automl["rolling_mean_growth"] = data_automl["y"].rolling(window=3).mean().diff().fillna(0)
        data_automl["sin_month"] = np.sin(2 * np.pi * data_automl["ds"].dt.month / 12)
        data_automl["cos_month"] = np.cos(2 * np.pi * data_automl["ds"].dt.month / 12)

        # Apply a log transform if necessary.
        if data_automl["y"].max() / data_automl["y"].min() > 5:
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
            time_budget = min(600, max(60, n * 0.1 + len(feature_cols) * 2))
            st.info(f"Dynamic time budget set to {time_budget} seconds based on dataset size and complexity.")

        eval_method = "cv" if len(X_train) >= 5 else "holdout"

        # Pass the custom dynamic metric to FLAML.
        automl_model = AutoML()
        automl_model.fit(
            X_train=X_train,
            y_train=y_train,
            task="regression",
            time_budget=time_budget,
            eval_method=eval_method,
            estimator_list=["xgboost", "lgbm", "rf", "catboost"],
            metric=dynamic_rmse_metric,   # <-- Using our custom dynamic metric here.
            early_stop=False,
            verbose=1
        )

        # Future Feature Generation (include our new peak features):
        future_features = []
        last_date = data_automl["ds"].iloc[-1]
        last_row = data_automl.iloc[-1].copy()

        for i in range(forecast_period):
            future_row = {}
            for lag in range(1, max_lag + 1):
                # For lag 1 use the last available value.
                if lag == 1:
                    value = last_row["y_log"] if apply_log else last_row["y"]
                    future_row[f"lag_{lag}"] = float(value)
                else:
                    prev_value = last_row.get(f"lag_{lag - 1}")
                    if prev_value is None:
                        prev_value = last_row["y_log"] if apply_log else last_row["y"]
                    future_row[f"lag_{lag}"] = float(prev_value)
            for window in [3, 6, 12]:
                lag_val = last_row.get(f"lag_{window}")
                if lag_val is not None:
                    if apply_log:
                        delta = (float(last_row["y_log"]) - float(lag_val)) / window
                    else:
                        delta = (float(last_row["y"]) - float(lag_val)) / window
                    base_val = last_row.get(f"rolling_mean_{window}", float(last_row["y_log"]) if apply_log else float(last_row["y"]))
                    future_row[f"rolling_mean_{window}"] = float(base_val) + delta
                else:
                    base_val = last_row.get(f"rolling_mean_{window}", float(last_row["y_log"]) if apply_log else float(last_row["y"]))
                    future_row[f"rolling_mean_{window}"] = float(base_val)
                std_val = last_row.get(f"rolling_std_{window}", 0.0)
                future_row[f"rolling_std_{window}"] = float(std_val)
                # New rolling max features: assume using last row's value as proxy.
                future_row[f"rolling_max_{window}"] = float(last_row.get(f"rolling_max_{window}", 
                                                                          float(last_row["y_log"]) if apply_log else float(last_row["y"])))
            # Propagate other engineered features.
            future_row["yoy_growth"] = float(last_row.get("yoy_growth", 0))
            future_row["y_diff"] = float(last_row.get("y_diff", 0))
            future_row["rolling_mean_growth"] = float(last_row.get("rolling_mean_growth", 0))
            future_month = (last_date.month + i) % 12 or 12
            future_row["sin_month"] = np.sin(2 * np.pi * future_month / 12)
            future_row["cos_month"] = np.cos(2 * np.pi * future_month / 12)
            # New: include a peak flag based on historical threshold.
            future_row["is_peak_month"] = 1 if future_month == data_automl["ds"].dt.month.mode()[0] else 0

            # Ensure no missing values remain.
            for key, value in future_row.items():
                if value is None:
                    future_row[key] = 0.0

            future_features.append(future_row)
            # Update last_row iteratively.
            last_row = last_row.copy()
            for key, value in future_row.items():
                last_row[key] = value

        future_df = pd.DataFrame(future_features)
        future_df.fillna(0, inplace=True)
        for col in X_train.columns:
            if col not in future_df.columns:
                future_df[col] = 0
        future_df = future_df[X_train.columns]

        automl_forecast = automl_model.predict(future_df)
        if automl_forecast is None:
            st.error("FLAML did not produce a valid model. Falling back to a naive forecast.")
            last_value = last_row["y_log"] if apply_log else last_row["y"]
            automl_forecast = np.full(forecast_period, float(last_value))
        if apply_log:
            automl_forecast = np.expm1(automl_forecast)

        forecast_df = pd.DataFrame({
            "ds": pd.date_range(start=last_date + pd.DateOffset(months=1),
                                 periods=forecast_period, freq="MS"),
            "yhat": automl_forecast,
            "yhat_lower": automl_forecast * 0.9,
            "yhat_upper": automl_forecast * 1.1
        })

        if isinstance(last_historical_value, (int, float)):
            forecast_df["yhat"] = inverse_difference(forecast_df["yhat"], last_historical_value)
            forecast_df["yhat_lower"] = inverse_difference(forecast_df["yhat_lower"], last_historical_value)
            forecast_df["yhat_upper"] = inverse_difference(forecast_df["yhat_upper"], last_historical_value)

        forecast_df = adjust_forecast(forecast_df, demand_shock, seasonality_adjustment, external_shock, category_scenarios)

        match_len = min(len(test["y"]), len(forecast_df))
        rmse = np.sqrt(mean_squared_error(test["y"].values[:match_len], forecast_df["yhat"].iloc[:match_len]))
        mape = mean_absolute_percentage_error(test["y"].values[:match_len], forecast_df["yhat"].iloc[:match_len])
        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast_df}

    except Exception as e:
        st.error(f"Updated AutoML Model failed: {e}")

    return "AutoML", result

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
    # Set theme toggle
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    theme = st.radio("🌙 Theme Mode:", ["Light", "Dark"], index=0 if st.session_state.theme=="light" else 1)
    st.session_state.theme = theme

    if st.session_state.theme=="dark":
        st.markdown("""
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
        st.markdown("""
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

    # Retrieve user info passed via query params
    query_params = st.experimental_get_query_params()
    user_id = query_params.get("user_id", ["guest"])[0]
    subscription_level = query_params.get("subscription_level", ["free"])[0]

    with st.sidebar:
        st.markdown("### 💡 Scenario Planning")
        if subscription_level != "premium":
            st.info("Scenario planning is available only for premium users.")
        else:
            st.success("You're using Premium. All features are enabled.")

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

            st.markdown("""
                <div style="text-align: center;">
                    <h2 style="color: #2B3A42;">🛠️ Map Your Columns</h2>
                </div>
                """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                date_column = st.selectbox("📅 Select the Date Column:", ["-- Select Column --"] + list(data.columns), key="date_col")
            with col2:
                sales_column = st.selectbox("💰 Select the Sales Column:", ["-- Select Column --"] + list(data.columns), key="sales_col")
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
                time_budget = 120
                forecast_period = 3
            else:
                st.markdown("### ⏱️ AutoML Time Budget")
                time_budget = st.slider("Set the time budget for AutoML training (in seconds):", min_value=60, max_value=1200, value=300, step=60)
                forecast_period = 24

            if date_column != "-- Select Column --" and sales_column != "-- Select Column --":
                last_date_in_data = data[date_column].max()
                min_future_date = (last_date_in_data + pd.DateOffset(days=1)).date()

                if subscription_level != "premium":
                    demand_shock = 0
                    seasonality_adjustment = 0
                    external_shock = False
                    category_scenarios = {}
                else:
                    st.sidebar.markdown("### 🎯 Scenario Planning")
                    demand_shock = st.sidebar.slider("Simulate Demand Shock (% Change in Sales):", -50, 50, 0, 5)
                    seasonality_adjustment = st.sidebar.slider("Adjust Seasonality Strength (% Change):", -50, 50, 0, 5)
                    external_shock = st.sidebar.checkbox("Simulate External Shock (e.g., Economic Downturn)")
                    category_scenarios = {}
                    if category_columns:
                        st.sidebar.markdown("### 🎯 Scenario Planning by Category (Dynamic)")
                        for col in category_columns:
                            st.sidebar.markdown(f"#### Adjustments for '{col}'")
                            unique_cats = sorted(data[col].dropna().unique())
                            selected_cats = st.sidebar.multiselect(f"Pick categories in '{col}' to adjust:", options=unique_cats)
                            category_scenarios[col] = {}
                            for cat in selected_cats:
                                with st.sidebar.expander(f"Adjust '{cat}' in '{col}'"):
                                    cat_adjust = st.slider(f"Percentage change for '{cat}'", -50, 50, 0, 5)
                                    cat_start = st.date_input(f"Start date for '{cat}'", value=min_future_date, min_value=min_future_date)
                                    default_end = (last_date_in_data + pd.DateOffset(months=1)).date()
                                    cat_end = st.date_input(f"End date for '{cat}'", value=default_end if default_end > min_future_date else min_future_date, min_value=cat_start)
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
                start_forecast = st.button("✅ Start Forecast", key="start_btn")
            else:
                start_forecast = st.button("⏳ Select Columns First", disabled=True, key="start_disabled")

            total_steps = 12 if subscription_level == "premium" else 6

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
                # overall_status.info(f"🔍 Last Historical Date: {last_historical_date}")
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
                    st.success("✅ Prophet Model Training Complete!")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

                    # STEP 6: Train ARIMA Model
                    step += 1
                    step_message.text(f"Step {step} of {total_steps}: Training ARIMA model...")
                    with st.spinner("🚀 Training ARIMA model..."):
                        arima_model_name, arima_res = train_arima_model(
                            train, test, forecast_period,
                            last_historical_value, is_diff,
                            demand_shock, seasonality_adjustment, external_shock, category_scenarios
                        )
                        time.sleep(1)
                    st.success("✅ ARIMA Model Training Complete!")
                    progress_bar.progress(int((step / total_steps) * 100))
                    time.sleep(1)

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
                        arima_model_name: arima_res,
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

                        # Save Forecast Data for Dashboard Access
                        
                        # Create folder if it doesn't exist
                        os.makedirs("forecast_data", exist_ok=True)

                        # Pick your user/session id
                        user_id = "user123"  # (Later tie this to real user/session)

                        # File path
                        forecast_json_path = f"forecast_data/forecast_{user_id}.json"

                        # Prepare the data (only date & forecasted value)
                        to_save = forecast_data[["ds", "yhat"]].copy()
                        to_save["ds"] = to_save["ds"].astype(str)  # ensure json serializable

                        with open(forecast_json_path, "w") as f:
                            json.dump(to_save.to_dict(orient="records"), f)

                        st.success("✅ Forecast saved successfully for Dashboard!")

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

                        forecast_df = st.session_state.model_results[best_model]["Forecast"]

                        csv_path = f"forecast_data/forecast_{user_id}.csv"
                        forecast_df.to_csv(csv_path, index=False)

                        if st.button("💾 Save Forecast to Dashboard"):
                            upload_forecast(
                                user_id=user_id,
                                forecast_name=f"My Forecast {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                                file_path=csv_path,
                                model_used=best_model,
                                time_horizon=len(forecast_df),
                                forecast_metrics={
                                    "rmse": st.session_state.model_results[best_model]["RMSE"],
                                    "mape": st.session_state.model_results[best_model]["MAPE"]
                                }
                            )
                            st.success("✅ Forecast Saved to Dashboard!")

                    except Exception as e:
                        st.error(f"❌ Error generating download file: {e}")

                else:
                    # FREE USERS PIPELINE (only AutoML)
                    # STEP 4 (Free): Train AutoML Model
                    st.markdown(
                        """
                        ⏱️ **AutoML Time Budget:**  
                        Free users are limited to a 60-second training time and a 3-month forecast window.  
                        
                        Upgrade to Premium to unlock extended time budgets (up to 20 minutes) and longer forecast horizons.
                        """
                    )
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