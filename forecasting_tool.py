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
        # "cap" is set to 20% above the max observed value and "floor" to 0.
        train["cap"] = 1.2 * train["y"].max()
        train["floor"] = 0
        
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
        # Add cap and floor to the future dataframe (using the same cap as training or adjust as needed)
        future["cap"] = train["cap"].max()
        future["floor"] = 0
        
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
                       demand_shock, seasonality_adjustment, external_shock, 
                       category_scenarios=None, time_budget=None):
    """
    Train an AutoML model using FLAML for time series forecasting with enhanced peak prediction.
    """
    result = {}
    try:
        # 1. Preprocessing: copy, ensure datetime, and sort
        data_automl = train.copy()
        data_automl["ds"] = pd.to_datetime(data_automl["ds"])
        data_automl = data_automl.sort_values("ds").reset_index(drop=True)

        # 2. Aggregate by date if category adjustments are enabled.
        if category_scenarios:
            data_automl = data_automl.groupby("ds", as_index=False).agg({"y": "sum"})

        n = len(data_automl)
        if n < 24:
            st.warning("Dataset has less than 24 records; seasonality patterns may not be reliably learned.")

        # 3. Determine maximum lag based on dataset size.
        max_lag = min(24, n - 1)
        if n <= 6:
            max_lag = min(3, n - 1)
        elif n <= 12:
            max_lag = min(6, n - 1)
        elif n <= 24:
            max_lag = min(12, n - 1)

        # 4. Add lag features.
        for lag in range(1, max_lag + 1):
            data_automl[f"lag_{lag}"] = data_automl["y"].shift(lag)

        # 5. Add rolling statistics (3- and 6-month windows).
        for window in [3, 6]:
            data_automl[f"rolling_mean_{window}"] = data_automl["y"].rolling(window=window, min_periods=1).mean()
            data_automl[f"rolling_std_{window}"] = data_automl["y"].rolling(window=window, min_periods=1).std()

        # 6. Add year-over-year growth.
        if n > 12:
            data_automl["yoy_growth"] = (data_automl["y"] / data_automl["y"].shift(12)) - 1
        else:
            data_automl["yoy_growth"] = 0

        # 7. Add differenced values and rolling mean growth.
        data_automl["y_diff"] = data_automl["y"].diff().fillna(0)
        data_automl["rolling_mean_growth"] = data_automl["y"].rolling(window=3).mean().diff().fillna(0)

        # 8. Add seasonal features: primary, second, and third harmonics.
        data_automl["sin_month"] = np.sin(2 * np.pi * data_automl["ds"].dt.month / 12)
        data_automl["cos_month"] = np.cos(2 * np.pi * data_automl["ds"].dt.month / 12)
        data_automl["sin2_month"] = np.sin(4 * np.pi * data_automl["ds"].dt.month / 12)
        data_automl["cos2_month"] = np.cos(4 * np.pi * data_automl["ds"].dt.month / 12)
        data_automl["sin3_month"] = np.sin(6 * np.pi * data_automl["ds"].dt.month / 12)
        data_automl["cos3_month"] = np.cos(6 * np.pi * data_automl["ds"].dt.month / 12)
        month_dummies = pd.get_dummies(data_automl["ds"].dt.month, prefix="month", drop_first=True)
        data_automl = pd.concat([data_automl, month_dummies], axis=1)

        # 9. Dynamic peak detection using recent 3 years.
        current_year = data_automl["ds"].dt.year.max()
        recent_years = 3
        recent_data = data_automl[data_automl["ds"].dt.year >= (current_year - recent_years + 1)]
        if recent_data.empty:
            recent_data = data_automl
        recent_monthly_avg = recent_data.groupby(recent_data["ds"].dt.month)["y"].mean()
        peak_month = recent_monthly_avg.idxmax()
        data_automl["is_peak_month"] = (data_automl["ds"].dt.month == peak_month).astype(int)
        data_automl["days_from_peak"] = (data_automl["ds"].dt.month - peak_month).apply(lambda x: x if x >= 0 else x + 12)

        # 10. Peak interaction features.
        data_automl["peak_sin"] = data_automl["is_peak_month"] * data_automl["sin_month"]
        data_automl["peak_cos"] = data_automl["is_peak_month"] * data_automl["cos_month"]
        if "lag_1" in data_automl.columns:
            data_automl["peak_lag1"] = data_automl["is_peak_month"] * data_automl["lag_1"]

        # 11. Sample weights for peak months (for info; not used in FLAML fit).
        data_automl["weight"] = data_automl["is_peak_month"].apply(lambda x: 2.0 if x else 1.0)

        # 12. Log transformation if necessary.
        if data_automl["y"].max() / data_automl["y"].min() > 5:
            data_automl["y_log"] = np.log1p(data_automl["y"])
            apply_log = True
        else:
            data_automl["y_log"] = data_automl["y"]
            apply_log = False

        data_automl.dropna(inplace=True)

        # 13. Prepare features and target.
        ignore_cols = ["y", "ds", "y_log", "weight"]
        feature_cols = [col for col in data_automl.columns if col not in ignore_cols]
        y_train = data_automl["y_log"] if apply_log else data_automl["y"]
        X_train = data_automl[feature_cols]

        # Preserve the last historical value before future feature generation.
        historical_last_value = data_automl.iloc[-1]["y_log"] if apply_log else data_automl.iloc[-1]["y"]

        # 14. Dynamic FLAML tuning.
        if n < 50:
            dynamic_metric = "mae"
            dynamic_estimators = ["xgboost", "rf"]
            dynamic_time_budget = min(300, max(60, n * 0.2 + len(feature_cols) * 2))
        elif n < 200:
            dynamic_metric = "rmse"
            dynamic_estimators = ["xgboost", "lgbm", "rf"]
            dynamic_time_budget = min(600, max(90, n * 0.15 + len(feature_cols) * 2))
        else:
            dynamic_metric = "rmse"
            dynamic_estimators = ["xgboost", "lgbm", "rf", "catboost"]
            dynamic_time_budget = min(600, max(120, n * 0.15 + len(feature_cols) * 2))
        if time_budget is not None:
            dynamic_time_budget = time_budget

        st.info(f"Dynamic settings: {dynamic_time_budget}s, metric: {dynamic_metric}, estimators: {dynamic_estimators}")
        eval_method = "cv" if len(X_train) >= 5 else "holdout"

        # 15. Train the FLAML AutoML model (without sample weights, as FLAML doesn't support them).
        automl_model = AutoML()
        automl_model.fit(
            X_train=X_train,
            y_train=y_train,
            task="regression",
            time_budget=dynamic_time_budget,
            eval_method=eval_method,
            estimator_list=dynamic_estimators,
            metric=dynamic_metric,
            verbose=1
        )

        # 16. Future feature generation (including seasonal and peak features).
        future_features = []
        last_date = data_automl["ds"].iloc[-1]
        last_row = data_automl.iloc[-1].copy()

        for i in range(forecast_period):
            future_row = {}
            # Lag features.
            for lag in range(1, max_lag + 1):
                if lag == 1:
                    value = last_row["y_log"] if apply_log else last_row["y"]
                    future_row[f"lag_{lag}"] = float(value)
                else:
                    prev_value = last_row.get(f"lag_{lag - 1}")
                    if prev_value is None:
                        prev_value = last_row["y_log"] if apply_log else last_row["y"]
                    future_row[f"lag_{lag}"] = float(prev_value)
            # Rolling statistics (windows 3 and 6).
            for window in [3, 6]:
                lag_val = last_row.get(f"lag_{window}")
                if lag_val is not None:
                    if apply_log:
                        delta = (float(last_row["y_log"]) - float(lag_val)) / window
                    else:
                        delta = (float(last_row["y"]) - float(lag_val)) / window
                    base_val = last_row.get(f"rolling_mean_{window}")
                    if base_val is None:
                        base_val = float(last_row["y_log"] if apply_log else last_row["y"])
                    future_row[f"rolling_mean_{window}"] = float(base_val) + delta
                else:
                    base_val = last_row.get(f"rolling_mean_{window}")
                    if base_val is None:
                        base_val = float(last_row["y_log"] if apply_log else last_row["y"])
                    future_row[f"rolling_mean_{window}"] = float(base_val)
                std_val = last_row.get(f"rolling_std_{window}")
                future_row[f"rolling_std_{window}"] = float(std_val) if std_val is not None else 0.0

            # Growth features.
            future_row["yoy_growth"] = float(last_row.get("yoy_growth", 0))
            future_row["y_diff"] = float(last_row.get("y_diff", 0))
            future_row["rolling_mean_growth"] = float(last_row.get("rolling_mean_growth", 0))
            
            # Seasonal features: Fourier terms and month dummies.
            future_month = (last_date.month + i) % 12 or 12
            future_row["sin_month"] = np.sin(2 * np.pi * future_month / 12)
            future_row["cos_month"] = np.cos(2 * np.pi * future_month / 12)
            future_row["sin2_month"] = np.sin(4 * np.pi * future_month / 12)
            future_row["cos2_month"] = np.cos(4 * np.pi * future_month / 12)
            future_row["sin3_month"] = np.sin(6 * np.pi * future_month / 12)
            future_row["cos3_month"] = np.cos(6 * np.pi * future_month / 12)
            month_dummies_future = {f"month_{m}": 1.0 if future_month == m else 0.0 for m in range(2, 13)}
            future_row.update(month_dummies_future)
            
            # Peak-related features.
            future_row["is_peak_month"] = 1.0 if future_month == peak_month else 0.0
            future_row["days_from_peak"] = (future_month - peak_month) if (future_month - peak_month) >= 0 else (future_month - peak_month + 12)
            future_row["peak_sin"] = future_row["is_peak_month"] * future_row["sin_month"]
            future_row["peak_cos"] = future_row["is_peak_month"] * future_row["cos_month"]
            if "lag_1" in feature_cols:
                future_row["peak_lag1"] = future_row["is_peak_month"] * future_row.get("lag_1", 0)
            
            # Ensure no None values.
            for key, value in future_row.items():
                if value is None:
                    future_row[key] = 0.0
            
            future_features.append(future_row)
            last_row = future_row.copy()  # Update last_row iteratively.

        future_df = pd.DataFrame(future_features).fillna(0)
        for col in X_train.columns:
            if col not in future_df.columns:
                future_df[col] = 0
        future_df = future_df[X_train.columns]

        # 17. Forecast generation.
        automl_forecast = automl_model.predict(future_df)
        if automl_forecast is None:
            st.error("FLAML did not produce a valid model. Falling back to a naive forecast.")
            automl_forecast = np.full(forecast_period, float(historical_last_value))
        if apply_log:
            automl_forecast = np.expm1(automl_forecast)

        forecast_df = pd.DataFrame({
            "ds": pd.date_range(start=last_date + pd.DateOffset(months=1),
                                 periods=forecast_period, freq="MS"),
            "yhat": automl_forecast
        })

        # 18. Inverse differencing if applicable.
        if isinstance(last_historical_value, (int, float)):
            forecast_df["yhat"] = inverse_difference(forecast_df["yhat"], last_historical_value)
        
        # 19. Apply forecast adjustments.
        forecast_df = adjust_forecast(forecast_df, demand_shock, seasonality_adjustment, external_shock, category_scenarios)

        # 20. Evaluate performance.
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