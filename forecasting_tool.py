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
                        is_diff, demand_shock, seasonality_adjustment, external_shock, category_scenarios=None):
    result = {}
    try:
        # If category adjustments are used, aggregate training data by date.
        if category_scenarios:
            train = train.groupby("ds", as_index=False).agg({"y": "sum"})
        
        # Set logistic growth parameters: 
        # "cap" is set to 20% above the max observed value and "floor" is set to 1 (instead of 0)
        train["cap"] = 1.2 * train["y"].max()
        train["floor"] = 1  # Setting a minimal positive floor to avoid negatives
        
        # Initialize Prophet with tuned parameters and logistic growth.
        model = Prophet(
            growth="logistic",  # enforce nonnegative forecasts
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
        # Add cap and floor to the future dataframe.
        future["cap"] = train["cap"].max()
        future["floor"] = 1
        
        forecast = model.predict(future)
        forecast = forecast[forecast["ds"] > train["ds"].max()]
        
        # Clamp forecasts to be nonnegative (if any negatives remain).
        forecast["yhat"] = forecast["yhat"].clip(lower=0)
        
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

def train_xgb_model(train, test, forecast_period, last_historical_value, is_diff,
                    demand_shock, seasonality_adjustment, external_shock, category_scenarios=None):
    result = {}
    try:
        # If category adjustments are used, aggregate training data by date.
        if category_scenarios:
            train = train.groupby("ds", as_index=False).agg({"y": "sum"})
        
        # 1. Detect historical patterns
        monthly_avg = train.groupby(train["ds"].dt.month)["y"].mean()
        peak_month = monthly_avg.idxmax()
        peak_value = monthly_avg.max()
        
        # 2. Feature engineering
        data_xgb = train.copy()
        data_xgb["month"] = data_xgb["ds"].dt.month  # Base feature
        data_xgb["year"] = data_xgb["ds"].dt.year
        data_xgb["month_year_interaction"] = data_xgb["month"] * (data_xgb["year"] - data_xgb["year"].min())
        
        # Peak alignment features
        data_xgb["days_from_peak"] = (data_xgb["month"] - peak_month).apply(lambda x: x if x >= 0 else x + 12)
        data_xgb["is_peak_month"] = (data_xgb["month"] == peak_month).astype(int)
        
        # Phase-aligned seasonal features
        phase_shift = peak_month - 1  # Shift waveform peak to historical peak month
        data_xgb["sin_month"] = np.sin(2 * np.pi * (data_xgb["month"] - phase_shift) / 12)
        data_xgb["cos_month"] = np.cos(2 * np.pi * (data_xgb["month"] - phase_shift) / 12)
        # Enhanced Fourier terms for more nuanced seasonality
        data_xgb["sin2_month"] = np.sin(4 * np.pi * (data_xgb["month"] - phase_shift) / 12)
        data_xgb["cos2_month"] = np.cos(4 * np.pi * (data_xgb["month"] - phase_shift) / 12)
        
        # Lag features with yearly focus
        max_lag = min(24, len(train) - 1)  # 2-year window
        for lag in [1, 2, 12, 24]:
            if lag <= max_lag:
                data_xgb[f"lag_{lag}"] = data_xgb["y"].shift(lag)
        
        data_xgb.dropna(inplace=True)
        
        # 3. Feature selection with fixed order
        feature_order = [
            'month', 'year', 'month_year_interaction', 'is_peak_month', 'days_from_peak',
            'sin_month', 'cos_month', 'sin2_month', 'cos2_month', 'lag_1', 'lag_2', 'lag_12', 'lag_24'
        ]
        feature_cols = [col for col in feature_order if col in data_xgb.columns]
    
        # 4. Model configuration
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        model.fit(data_xgb[feature_cols], data_xgb["y"])
        
        # 5. Compute base dynamic boost factor based on recent peaks (using the last 3 years)
        recent_years = sorted(train["ds"].dt.year.unique())[-3:]
        recent_data = train[train["ds"].dt.year.isin(recent_years)]
        recent_monthly_avg = recent_data.groupby(recent_data["ds"].dt.month)["y"].mean()
        
        if recent_monthly_avg.mean() != 0:
            dynamic_boost = recent_monthly_avg.get(peak_month, peak_value) / recent_monthly_avg.mean()
        else:
            dynamic_boost = 1.25
        
        # Optionally, clamp the base boost factor to a reasonable range
        dynamic_boost = max(min(dynamic_boost, 1.2), 1.0)
        
        # 6. Compute YOY growth for peak values using the actual peak month values per year
        yearly_peaks = {}
        for year in sorted(train["ds"].dt.year.unique()):
            year_data = train[(train["ds"].dt.year == year) & (train["ds"].dt.month == peak_month)]
            if not year_data.empty:
                yearly_peaks[year] = year_data["y"].mean()  # or .max() if you prefer
        
        growth_factors = []
        unique_years = sorted(yearly_peaks.keys())
        for i in range(len(unique_years) - 1):
            y1 = unique_years[i]
            y2 = unique_years[i+1]
            if yearly_peaks[y1] > 0:
                growth_factors.append(yearly_peaks[y2] / yearly_peaks[y1])
        if growth_factors:
            yoy_growth = np.mean(growth_factors)
        else:
            yoy_growth = 1.0

        # Impose a floor to ensure peaks do not decrease (if your data supports growth)
        yoy_growth = max(yoy_growth, 1.0)
        
        # Remember the last training year for compounding
        last_training_year = unique_years[-1]
        
        # 7. Forecasting with dynamic, compounding peak adjustment
        last_row = data_xgb[feature_cols].iloc[-1].copy()
        last_date = data_xgb["ds"].iloc[-1]
        preds = []
        
        for i in range(forecast_period):
            future_date = last_date + pd.DateOffset(months=i+1)
            is_peak = int(future_date.month == peak_month)
            future_year = future_date.year
            
            # Compute compounded boost for future year:
            years_ahead = future_year - last_training_year
            compounded_boost = dynamic_boost * (yoy_growth ** years_ahead)
            compounded_boost = max(compounded_boost, 1.0)  # ensure it doesn't fall below 1.0
            
            # Initialize features for the future date
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
            
            # Predict with the XGBoost model
            base_pred = model.predict(pd.DataFrame([features]))[0]
            
            # Apply the compounded boost for the peak month
            if is_peak:
                pred = base_pred * compounded_boost
            else:
                pred = base_pred
                
            preds.append(pred)
            
            # Update state for lag features; shift previous lags and use current prediction as new lag_1
            for lag in [24, 12, 2, 1]:
                if f'lag_{lag}' in feature_cols:
                    if lag == 1:
                        last_row['lag_1'] = pred
                    else:
                        last_row[f'lag_{lag}'] = last_row.get(f'lag_{lag-1}', pred)
        
        # 8. Create forecast dataframe
        forecast_df = pd.DataFrame({
            "ds": pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_period,
                freq="MS"
            ),
            "yhat": preds
        })
    
        # 9. Post-processing
        if is_diff and last_historical_value is not None:
            forecast_df["yhat"] = last_historical_value + forecast_df["yhat"].cumsum()
            
        forecast_df = adjust_forecast(forecast_df, demand_shock, seasonality_adjustment, external_shock, category_scenarios)
    
        # 10. Validation
        match_len = min(len(test), len(forecast_df))
        if match_len > 0:
            rmse = mean_squared_error(test["y"].iloc[:match_len], forecast_df["yhat"].iloc[:match_len], squared=False)
            mape = mean_absolute_percentage_error(test["y"].iloc[:match_len], forecast_df["yhat"].iloc[:match_len])
        else:
            rmse, mape = np.nan, np.nan
    
        # 11. Robust diagnostics
        st.write(f"📅 Historical Peak: {calendar.month_abbr[peak_month]}")
        if not forecast_df.empty:
            try:
                peak_idx = forecast_df["yhat"].idxmax()
                if 0 <= peak_idx < len(forecast_df):
                    fcst_peak_month = forecast_df.iloc[peak_idx]["ds"].month
                    st.write(f"🔮 Forecast Peak: {calendar.month_abbr[fcst_peak_month]}")
                else:
                    st.warning("⚠️ Could not determine forecast peak")
            except Exception as e:
                st.warning("⚠️ Peak detection failed")
    
        result = {"RMSE": float(rmse), "MAPE": float(mape), "Forecast": forecast_df}
    
    except Exception as e:
        st.warning(f"❌ XGBoost Model Failed: {str(e)}")
        result = {"error": str(e)}
    
    return "XGBoost", result

def train_automl_model(train, test, forecast_period, last_historical_value, is_diff, 
                       demand_shock, seasonality_adjustment, external_shock, category_scenarios=None, time_budget=None):
    """
    Train an AutoML model using FLAML for time series forecasting.
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

        # Decide on evaluation method based on number of samples:
        # Use holdout if we have too few samples for 5-fold CV.
        eval_method = "cv" if len(X_train) >= 5 else "holdout"

        # Train AutoML model using FLAML
        automl_model = AutoML()
        automl_model.fit(
            X_train=X_train,
            y_train=y_train,
            task="regression",
            time_budget=time_budget,
            eval_method=eval_method,
            estimator_list=["xgboost", "lgbm", "rf", "catboost"],
            metric="r2",
            early_stop=False,
            verbose=1
        )

        # Generate future features for forecasting
        future_features = []
        last_date = data_automl["ds"].iloc[-1]
        last_row = data_automl.iloc[-1].copy()

        for i in range(forecast_period):
            future_row = {}
            # Update lag features based on available max_lag
            for lag in range(1, max_lag + 1):
                if lag == 1:
                    value = last_row["y_log"] if apply_log else last_row["y"]
                    future_row[f"lag_{lag}"] = float(value)
                else:
                    prev_value = last_row.get(f"lag_{lag - 1}")
                    if prev_value is None:
                        prev_value = last_row["y_log"] if apply_log else last_row["y"]
                    future_row[f"lag_{lag}"] = float(prev_value)
            # Update rolling statistics for each window
            for window in [3, 6, 12]:
                lag_val = last_row.get(f"lag_{window}")
                if lag_val is not None:
                    if apply_log:
                        delta = (float(last_row["y_log"]) - float(lag_val)) / window
                    else:
                        delta = (float(last_row["y"]) - float(lag_val)) / window
                    base_val = last_row.get(f"rolling_mean_{window}")
                    if base_val is None:
                        base_val = last_row["y_log"] if apply_log else last_row["y"]
                    future_row[f"rolling_mean_{window}"] = float(base_val) + delta
                else:
                    base_val = last_row.get(f"rolling_mean_{window}")
                    if base_val is None:
                        base_val = last_row["y_log"] if apply_log else last_row["y"]
                    future_row[f"rolling_mean_{window}"] = float(base_val)
                std_val = last_row.get(f"rolling_std_{window}")
                future_row[f"rolling_std_{window}"] = float(std_val) if std_val is not None else 0.0

            future_row["yoy_growth"] = float(last_row.get("yoy_growth", 0))
            future_row["y_diff"] = float(last_row.get("y_diff", 0))
            future_row["rolling_mean_growth"] = float(last_row.get("rolling_mean_growth", 0))
            
            # Update trigonometric features based on the future month
            future_month = (last_date.month + i) % 12 or 12
            future_row["sin_month"] = np.sin(2 * np.pi * future_month / 12)
            future_row["cos_month"] = np.cos(2 * np.pi * future_month / 12)
            
            # Ensure no None values remain in future_row
            for key, value in future_row.items():
                if value is None:
                    future_row[key] = 0.0
            
            future_features.append(future_row)
            
            # Update last_row with future_row values for iterative forecasting
            last_row = last_row.copy()
            for key, value in future_row.items():
                last_row[key] = value

        # Create future DataFrame ensuring all required columns are present
        future_df = pd.DataFrame(future_features)
        future_df.fillna(0, inplace=True)
        for col in X_train.columns:
            if col not in future_df.columns:
                future_df[col] = 0
        future_df = future_df[X_train.columns]

        # Generate forecasts
        automl_forecast = automl_model.predict(future_df)
        # Fallback: if predict returns None, use a naive forecast (repeat last value)
        if automl_forecast is None:
            st.error("FLAML did not produce a valid model. Falling back to a naive forecast.")
            last_value = last_row["y_log"] if apply_log else last_row["y"]
            automl_forecast = np.full(forecast_period, float(last_value))

        if apply_log:
            automl_forecast = np.expm1(automl_forecast)

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            "ds": pd.date_range(start=last_date + pd.DateOffset(months=1),
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
    subscription_level = "premium"  # Change as needed

    if subscription_level != "premium":
        st.info("You are using the Free version. Advanced features are disabled.")

    # Reset Button in Sidebar
    if st.sidebar.button("🔄 Reset App"):
        st.session_state.clear()
        st.experimental_rerun()

    # Sidebar: Scenario Planning and AutoML Time Budget (only for premium users)
    if subscription_level == "premium":
        st.sidebar.markdown("### 🎯 Scenario Planning")
        demand_shock = st.sidebar.slider(
            "Simulate Demand Shock (% Change in Sales):",
            min_value=-50, max_value=50, value=0, step=5
        )
        seasonality_adjustment = st.sidebar.slider(
            "Adjust Seasonality Strength (% Change):",
            min_value=-50, max_value=50, value=0, step=5
        )
        external_shock = st.sidebar.checkbox(
            "Simulate External Shock (e.g., Economic Downturn)"
        )
        category_scenarios = {}
        # (You can implement category-specific scenario planning here if needed.)
        st.sidebar.markdown("### ⏱️ AutoML Time Budget")
        time_budget = st.sidebar.slider(
            "Set AutoML time budget (sec):", min_value=60, max_value=1200, value=300, step=60
        )
        forecast_period = 24
    else:
        st.sidebar.info("Upgrade to Premium for Scenario Planning.")
        demand_shock = 0
        seasonality_adjustment = 0
        external_shock = False
        category_scenarios = {}
        time_budget = 60
        forecast_period = 3

    # --------------------- Tab Layout ---------------------
    tab1, tab2, tab3 = st.tabs(["Data & Setup", "Model Training & Forecast", "Results & Insights"])

    # --------- Tab 1: Data & Setup ---------
    with tab1:
        st.header("📥 Upload and Setup Your Data")
        uploaded_file = st.file_uploader("Upload your sales data file (CSV)", type=["csv"])
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                st.subheader("Uploaded Data Preview")
                st.dataframe(data.head())
                st.markdown("### Map Your Columns")
                col1, col2, col3 = st.columns(3)
                with col1:
                    date_column = st.selectbox(
                        "📅 Select the Date Column:",
                        ["-- Select Column --"] + list(data.columns), key="date_col"
                    )
                with col2:
                    sales_column = st.selectbox(
                        "💰 Select the Sales Column:",
                        ["-- Select Column --"] + list(data.columns), key="sales_col"
                    )
                with col3:
                    if date_column == "-- Select Column --" or sales_column == "-- Select Column --":
                        st.warning("Please select both Date and Sales columns.")
                        category_columns = None
                    else:
                        if subscription_level != "premium":
                            st.info("Category adjustments available only for Premium users.")
                            category_columns = None
                        else:
                            category_columns = st.multiselect(
                                "🏷️ Select Category Columns (Optional):",
                                options=[col for col in data.columns if col not in [date_column, sales_column]],
                                key="category_cols"
                            )
                if date_column != "-- Select Column --":
                    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
                st.session_state["raw_data"] = data.copy()
            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            st.info("Please upload a CSV file to begin.")

    # --------- Tab 2: Model Training & Forecast ---------
    with tab2:
        st.header("🚀 Model Training and Forecasting")
        if "raw_data" in st.session_state and st.session_state["raw_data"] is not None:
            data = st.session_state["raw_data"]
            if st.session_state.get("date_col") and st.session_state.get("sales_col") and st.session_state["raw_data"] is not None:
                with st.spinner("Preprocessing data..."):
                    processed_data, last_historical_value, y_original, is_diff = preprocess_data(
                        data, st.session_state.date_col, st.session_state.sales_col, st.session_state.get("category_cols")
                    )
                    time.sleep(1)
                if processed_data is None:
                    st.error("Preprocessing failed.")
                else:
                    st.success("Data Preprocessed Successfully!")
                    st.subheader("Processed Monthly Data")
                    with st.expander("View Processed Data"):
                        st.dataframe(processed_data)
                    testing_period = int(len(processed_data) * 0.2)
                    train = processed_data.iloc[:-testing_period]
                    test = processed_data.iloc[-testing_period:]
                    progress_bar = st.progress(0)
                    step_message = st.empty()

                    # For brevity, focusing on AutoML; integrate other models similarly if desired.
                    step_message.text("Training AutoML model...")
                    with st.spinner("Training AutoML..."):
                        model_name, automl_res = train_automl_model(
                            train, test, forecast_period,
                            last_historical_value, is_diff,
                            demand_shock, seasonality_adjustment, external_shock, category_scenarios, time_budget
                        )
                        time.sleep(1)
                    st.success("AutoML Model Training Complete!")
                    st.session_state.model_results = {"AutoML": automl_res}
                    progress_bar.progress(100)
                    step_message.text("Training steps completed!")
            else:
                st.warning("Please complete column mapping in the 'Data & Setup' tab.")
        else:
            st.info("Upload your data in the 'Data & Setup' tab to proceed.")

    # --------- Tab 3: Results & Insights ---------
    with tab3:
        st.header("🔍 Forecast Results and Insights")
        if "model_results" in st.session_state:
            results = st.session_state.model_results
            automl_result = results.get("AutoML", {})
            forecast_df = automl_result.get("Forecast")
            if forecast_df is not None:
                st.subheader("Forecast Metrics")
                st.write(f"**RMSE:** {automl_result.get('RMSE', 'N/A'):.2f}")
                st.write(f"**MAPE:** {automl_result.get('MAPE', 'N/A'):.2%}")
                st.subheader("Forecast Visualization")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_original["ds"],
                    y=y_original["y_original"],
                    mode="lines",
                    name="Historical Data",
                    line=dict(color="black", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df["ds"],
                    y=forecast_df["yhat"],
                    mode="lines",
                    name="AutoML Forecast",
                    line=dict(color="purple", width=2)
                ))
                fig.update_layout(
                    title="📊 Sales Forecast",
                    xaxis_title="Date",
                    yaxis_title="Sales",
                    template="plotly_white",
                    xaxis_tickformat="%Y-%m"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("### Download Forecast Data")
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📩 Download Forecast CSV",
                    data=csv,
                    file_name="forecast.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No valid forecast results available.")
        else:
            st.info("Train a model in the 'Model Training & Forecast' tab to see results.")

if __name__ == "__main__":
    main()
