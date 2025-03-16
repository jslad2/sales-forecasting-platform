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

# Enable Wide Mode (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(layout="wide", page_title="Time Series Forecasting", page_icon="📈")

# Add dark mode toggle
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Theme toggle
theme = st.radio("🌙 Theme Mode:", ["Light", "Dark"], index=0 if st.session_state.theme == "light" else 1)
st.session_state.theme = theme

# Apply the selected theme
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
    """
    Perform the Augmented Dickey-Fuller (ADF) and KPSS tests to check stationarity.
    """
    # ADF Test
    adf_result = adfuller(series, autolag="AIC")
    adf_p_value = adf_result[1]

    # KPSS Test
    kpss_result = kpss(series, regression="c", nlags="auto")
    kpss_p_value = kpss_result[1]

    # Determine stationarity
    if adf_p_value < 0.05 and kpss_p_value > 0.05:
        return "Stationary"
    elif adf_p_value >= 0.05 and kpss_p_value <= 0.05:
        return "Non-Stationary"
    else:
        return "Inconclusive"

def preprocess_data(data, date_column, sales_column):
    """
    Preprocess the uploaded data, check stationarity, and apply transformations if needed.
    Returns a dataframe with columns "ds" and "y" for Prophet compatibility.
    Also returns the last historical value before forecasting if differencing is applied.
    """
    try:
        # Convert date column to datetime
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data.dropna(subset=[date_column, sales_column], inplace=True)

        # Aggregate to Monthly Data
        data = data[[date_column, sales_column]].rename(columns={date_column: "ds", sales_column: "y"})
        data["ds"] = pd.to_datetime(data["ds"])
        data = data.groupby(data["ds"].dt.to_period("M")).agg({"y": "sum"}).reset_index()
        data["ds"] = data["ds"].dt.to_timestamp()

        # Store the original values
        y_original = data[["ds", "y"]].copy().rename(columns={"y": "y_original"})

        # Check stationarity
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

        # Apply differencing if non-stationary
        if stationarity_result == "Non-Stationary":
            st.warning("Applying differencing to stabilize the series.")
            data["y_diff"] = data["y"].diff()
            last_historical_value = data["y"].iloc[-1]

            # Create a Plotly figure for visualization
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
                title="Original vs Differenced Series",
                xaxis_title="Date",
                yaxis_title="Sales",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Remove NaNs caused by differencing
            differenced_data = data.dropna(subset=["y_diff"]).rename(columns={"y_diff": "y"})
            return differenced_data[["ds", "y"]], last_historical_value, y_original

        else:
            # If stationary, plot only the original series
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["ds"],
                y=data["y"],
                mode="lines",
                name="Original Series",
                line=dict(color="blue", width=2)
            ))
            fig.update_layout(
                title="📊 Original Series",
                xaxis_title="Date",
                yaxis_title="Sales",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Return original data and None for last_historical_value (no differencing applied)
            return data[["ds", "y"]], None, y_original

    except Exception as e:
        st.error(f"An error occurred during preprocessing: {e}")
        return None, None, None  # Return None in case of an error

def inverse_difference(forecast_data, first_value):
    """Reverse differencing to restore original scale."""
    if first_value is not None:
        # Ensure first_value is numeric
        if not isinstance(first_value, (int, float)):
            raise ValueError("first_value must be a numeric value (int or float).")

        # Restore original values using cumulative sums
        forecast_data["yhat"] = first_value + forecast_data["yhat"].cumsum().shift(fill_value=first_value)
        forecast_data["yhat_upper"] = first_value + forecast_data["yhat_upper"].cumsum().shift(fill_value=first_value)
        forecast_data["yhat_lower"] = first_value + forecast_data["yhat_lower"].cumsum().shift(fill_value=first_value)
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

    best_params = None
    best_rmse = float("inf")

    if dataset_length < 100:
        horizon_days = min(7, max(3, dataset_length // 5))
        initial_days = max(30, dataset_length // 2)
    else:
        horizon_days = min(30, max(7, dataset_length // 10))
        initial_days = max(90, dataset_length // 3)

    horizon = f"{horizon_days} days"
    initial = f"{initial_days} days"
    period = f"{horizon_days // 2} days"

    for params in tqdm(ParameterGrid(param_grid), desc="Hyperparameter Search"):
        try:
            prophet_model = Prophet(
                seasonality_mode=params["seasonality_mode"],
                changepoint_prior_scale=params["changepoint_prior_scale"],
                yearly_seasonality=dataset_length >= 365,
                weekly_seasonality=dataset_length >= 30,
                daily_seasonality=False
            )

            prophet_model.fit(train)

            cv_results = cross_validation(
                prophet_model,
                initial=initial,
                horizon=horizon,
                period=period
            )
            metrics = performance_metrics(cv_results)

            rmse = metrics["rmse"].mean()

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params

        except Exception as e:
            st.write(f"Failed with params {params}: {e}")
            continue

    return best_params, best_rmse

def apply_scenarios(data, demand_shock, seasonality_adjustment, external_shock):
    """
    Apply scenario adjustments to the training data.
    """
    # Apply demand shock
    if demand_shock != 0:
        data["y"] = data["y"] * (1 + demand_shock / 100)

    # Apply seasonality adjustment
    if seasonality_adjustment != 0:
        # Example: Adjust monthly seasonality
        data["month"] = data["ds"].dt.month
        seasonality_multiplier = 1 + seasonality_adjustment / 100
        data["y"] = data["y"] * (1 + (data["month"] - 1) * (seasonality_multiplier - 1) / 12)

    # Apply external shock (e.g., economic downturn)
    if external_shock:
        data["y"] = data["y"] * 0.8  # Simulate a 20% reduction in sales

    # Return the modified data
    return data

def main():
    user_id = "user123"
    subscription_level = "premium"

    if subscription_level != "premium":
        st.warning("🔒 Upgrade to Premium to unlock advanced features like AutoML, scenario planning, and more!")
        st.stop()

    # File Upload
    uploaded_file = st.file_uploader("Upload your sales data file", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(data)

            # 📌 Center "Map Your Columns" Section
            st.markdown(
                """
                <div style="text-align: center;">
                    <h2 style="color: #2B3A42;">🛠️ Map Your Columns</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 🗂 Dropdowns for Column Selection (Centered)
            col1, col2 = st.columns([1, 1])
            with col1:
                date_column = st.selectbox("📅 Select the Date Column:", ["-- Select Column --"] + list(data.columns), key="date_col")
            with col2:
                sales_column = st.selectbox("💰 Select the Sales Column:", ["-- Select Column --"] + list(data.columns), key="sales_col")

            # Scenario Planning Section
            st.sidebar.markdown("### 🎯 Scenario Planning")

            # Demand Shock Scenario
            demand_shock = st.sidebar.slider(
                "Simulate Demand Shock (% Change in Sales):",
                min_value=-50, max_value=50, value=0, step=5
            )

            # Seasonality Adjustment
            seasonality_adjustment = st.sidebar.slider(
                "Adjust Seasonality Strength (% Change):",
                min_value=-50, max_value=50, value=0, step=5
            )

            # External Shock (e.g., Economic Downturn)
            external_shock = st.sidebar.checkbox(
                "Simulate External Shock (e.g., Economic Downturn)"
            )

            # 🚀 Disable "Start Forecast" Button Until Valid Selections
            if date_column != "-- Select Column --" and sales_column != "-- Select Column --":
                start_forecast = st.button("✅ Start Forecast", key="start_btn", help="Click to generate your AI-powered forecast")
            else:
                start_forecast = st.button("⏳ Select Columns First", disabled=True, key="start_disabled")

            # 🏁 Run Forecast Only If Button is Clicked
            if start_forecast:
                # Preprocess Data
                processed_data, last_historical_value, y_original = preprocess_data(data, date_column, sales_column)
                if processed_data is None:  # Check if preprocessing failed
                    st.error("❌ Preprocessing failed. Please check your data and try again.")
                    return

                # Dynamically determine the last historical date from y_original
                last_historical_date = y_original["ds"].max()
                st.write(f"🔍 Last Historical Date: {last_historical_date}")

                # Apply scenarios to training data
                scenario_data = apply_scenarios(processed_data.copy(), demand_shock, seasonality_adjustment, external_shock)

                # ✅ Centered Header with Icon
                st.markdown(
                    """
                    <div style="text-align: center;">
                        <h2 style="color: #2B3A42; font-size: 1.8em;">
                            📅 Preprocessed Monthly Data
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ✅ Use Streamlit's Expander to Organize Data
                with st.expander("📊 View Processed Data"):
                    # ✅ Adjust Column Widths Dynamically
                    st.markdown(
                        """
                        <style>
                        .stDataFrame { text-align: center; margin: auto; }
                        .stDataFrame table { width: 100% !important; }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ✅ Display DataFrame with Improved Spacing
                    st.dataframe(
                        scenario_data.style.set_properties(**{"text-align": "center"}),
                        width=1400,  # Wider Table
                        height=450   # Show More Rows
                    )

                # Determine Testing Period Dynamically
                testing_period = int(len(scenario_data) * 0.2)
                train = scenario_data.iloc[:-testing_period]
                test = scenario_data.iloc[-testing_period:]

                st.write(f"🔍 Train DataFrame: {train}")

                forecast_period = 24  # Fixed to 12 months forecast

                # Forecasting Models
                results = {}

                st.write("🚀 Finding the best Prophet hyperparameters...")
                best_params, best_rmse = find_best_prophet_params(train)

                if best_params is None:
                    st.error("❌ No valid Prophet parameters were found. Check your data preprocessing or parameter grid.")
                    return

                st.write(f"✅ Best Parameters: {best_params}")
                st.write(f"📉 Best RMSE from cross-validation: {best_rmse}")

                try:
                    st.write("📊 Training Prophet Model with Best Parameters...")

                    # Ensure training data is valid
                    if train.shape[0] == 0:
                        st.error("🚨 Training data is empty! Check preprocessing.")
                        raise ValueError("Train dataset has no rows.")

                    # Define Prophet model
                    prophet_model = Prophet(
                        seasonality_mode=best_params["seasonality_mode"],
                        changepoint_prior_scale=best_params["changepoint_prior_scale"]
                    )

                    # Detect & dynamically add seasonalities
                    try:
                        prophet_model = detect_and_add_seasonalities(prophet_model, train)
                    except Exception as e:
                        st.warning(f"⚠️ Seasonality detection failed: {e}. Proceeding without additional seasonalities.")

                    # Debugging train data
                    st.write("🔍 Train DataFrame Sample:")
                    st.dataframe(train.head())
                    st.write(f"🔍 Train DataFrame Shape: {train.shape}")
                    st.write("🧐 Min Date in Train:", train["ds"].min())
                    st.write("🧐 Max Date in Train:", train["ds"].max())

                    # Train the Prophet model
                    prophet_model.fit(train)

                    # Generate future dates & predict
                    future = prophet_model.make_future_dataframe(
                        periods=forecast_period,
                        freq="M",
                        include_history=False  # Exclude historical data
                    )
                    prophet_forecast = prophet_model.predict(future)

                    # Ensure only future forecasts are used
                    prophet_forecast = prophet_forecast[prophet_forecast["ds"] > train["ds"].max()]

                    # Debug forecast output
                    st.write("🔍 Future Forecast DataFrame (Filtered):")
                    st.dataframe(prophet_forecast.head())

                    # Ensure test set matches forecast length for metric calculation
                    matching_length = min(len(test["y"]), len(prophet_forecast))
                    prophet_rmse = mean_squared_error(
                        test["y"].iloc[:matching_length], prophet_forecast["yhat"].iloc[:matching_length]
                    ) ** 0.5
                    prophet_mape = mean_absolute_percentage_error(
                        test["y"].iloc[:matching_length], prophet_forecast["yhat"].iloc[:matching_length]
                    )

                    # Restore differenced values if applied
                    if isinstance(last_historical_value, (int, float)):  # Ensure it's numeric
                        prophet_forecast = inverse_difference(prophet_forecast, last_historical_value)

                    # Debugging: Check first forecasted value
                    st.write("✅ First Forecasted Value After Processing:", prophet_forecast["yhat"].iloc[0])

                    # Identify highest & lowest forecasted sales
                    highest_point = prophet_forecast.loc[prophet_forecast["yhat"].idxmax()]
                    lowest_point = prophet_forecast.loc[prophet_forecast["yhat"].idxmin()]

                    # Display key insights
                    summary_text = (
                        f"### 📊 Key Insights\n"
                        f"- **Projected Growth:** Sales are expected to {'increase' if prophet_forecast['yhat'].iloc[-1] > test['y'].iloc[-1] else 'decrease'} "
                        f"by {abs(((prophet_forecast['yhat'].iloc[-1] - test['y'].iloc[-1]) / test['y'].iloc[-1]) * 100):.2f}% in the next period.\n"
                        f"- **Highest Predicted Sales:** {highest_point['yhat']:.2f} on {highest_point['ds'].strftime('%Y-%m-%d')}\n"
                        f"- **Lowest Predicted Sales:** {lowest_point['yhat']:.2f} on {lowest_point['ds'].strftime('%Y-%m-%d')}\n"
                        f"- **Best Model Parameters:** {best_params}\n"
                        f"- **Performance Metrics:**\n"
                        f"  - 📉 RMSE: {prophet_rmse:.2f}\n"
                        f"  - 📉 MAPE: {prophet_mape:.2f}\n"
                    )

                    with st.expander("📊 Prophet Model Summary"):
                        st.markdown(summary_text)

                        # Ensure train["ds"].max() is valid
                    st.write("🧐 Max Date in Train (Before Plotly):", train["ds"].max())

                    # Create the forecast visualization
                    fig = go.Figure()

                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=y_original["ds"],
                        y=y_original["y_original"],
                        mode="lines",
                        name="Historical",
                        line=dict(color="black", width=2)
                    ))

                    # Add forecasted data
                    fig.add_trace(go.Scatter(
                        x=prophet_forecast["ds"],
                        y=prophet_forecast["yhat"],
                        mode="lines",
                        name="Forecast",
                        line=dict(color="blue", width=2)
                    ))

                    # Add confidence intervals
                    fig.add_trace(go.Scatter(
                        x=prophet_forecast["ds"],
                        y=prophet_forecast["yhat_upper"],
                        mode="lines",
                        name="Upper Confidence",
                        line=dict(color="lightblue", dash="dot")
                    ))
                    fig.add_trace(go.Scatter(
                        x=prophet_forecast["ds"],
                        y=prophet_forecast["yhat_lower"],
                        mode="lines",
                        name="Lower Confidence",
                        line=dict(color="lightblue", dash="dot")
                    ))

                    # Update layout
                    fig.update_layout(
                        title="Prophet Forecast with Confidence Intervals",
                        xaxis_title="Date",
                        yaxis_title="Sales",
                        legend_title="Legend",
                        template="plotly_white"
                    )

                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)

                    # Save results
                    results["Prophet"] = {
                        "RMSE": float(prophet_rmse),
                        "MAPE": float(prophet_mape),
                        "Forecast": prophet_forecast
                    }

                except Exception as e:
                    st.warning(f"❌ Prophet Model failed: {e}")
                    
                # ARIMA Model
                st.write("🔄 Training ARIMA Model...")

                try:
                    # Detect seasonality using seasonal decomposition & ACF values
                    st.write("📊 Analyzing seasonality in the data...")
                    try:
                        decomposition = seasonal_decompose(train["y"], model="additive", period=12)
                        seasonality_present = np.any(np.abs(decomposition.seasonal) > 0.01)

                        # Compute ACF values instead of using plot_acf
                        acf_values = acf(train["y"], nlags=12, fft=False)
                        seasonality_confirmed = any(np.abs(acf_values[1:]) > 0.2)  # Ignore lag 0

                        if seasonality_present and seasonality_confirmed:
                            st.write("✅ Seasonality detected. Using **seasonal ARIMA**.")
                            seasonal = True
                        else:
                            st.write("⚠️ No significant seasonality detected. Using **non-seasonal ARIMA**.")
                            seasonal = False
                    except Exception as e:
                        st.warning(f"⚠️ Error in seasonality analysis: {e}")
                        seasonal = False  # Default to non-seasonal ARIMA if error occurs

                    # Automatically configure ARIMA model
                    arima_model = auto_arima(
                        train["y"],
                        seasonal=seasonal,
                        m=12 if seasonal else 1,
                        d=None,  # Auto-detect differencing
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

                    # Generate future forecast
                    st.write("📅 Generating ARIMA Forecast...")
                    arima_forecast = arima_model.predict(n_periods=forecast_period)

                    # Ensure forecast matches expected length
                    if len(arima_forecast) < forecast_period:
                        st.warning("⚠️ ARIMA forecast shorter than expected. Adjusting to match test set length.")
                        forecast_period = len(arima_forecast)

                    # Prepare forecast DataFrame
                    forecast_dates = pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1), periods=forecast_period, freq="M")
                    forecast_df = pd.DataFrame({"ds": forecast_dates, "yhat": arima_forecast[:forecast_period]})

                    # Evaluate performance
                    matching_length = min(len(test["y"]), len(forecast_df))
                    test_y_trimmed = test["y"].iloc[:matching_length]
                    forecast_y_trimmed = forecast_df["yhat"].iloc[:matching_length]

                    arima_rmse = mean_squared_error(test_y_trimmed, forecast_y_trimmed) ** 0.5
                    arima_mape = mean_absolute_percentage_error(test_y_trimmed, forecast_y_trimmed)

                    # Identify high and low points
                    highest_point = forecast_df.loc[forecast_df["yhat"].idxmax()]
                    lowest_point = forecast_df.loc[forecast_df["yhat"].idxmin()]

                    # Display Key Insights
                    summary_text = (
                        f"### Key Insights\n"
                        f"- **Projected Growth:** Sales are expected to {'increase' if forecast_df['yhat'].iloc[-1] > test['y'].iloc[-1] else 'decrease'} "
                        f"by {abs(((forecast_df['yhat'].iloc[-1] - test['y'].iloc[-1]) / test['y'].iloc[-1]) * 100):.2f}% in the next period.\n"
                        f"- **Highest Predicted Sales:** {highest_point['yhat']:.2f} on {highest_point['ds'].strftime('%Y-%m-%d')}\n"
                        f"- **Lowest Predicted Sales:** {lowest_point['yhat']:.2f} on {lowest_point['ds'].strftime('%Y-%m-%d')}\n"
                        f"- **Performance Metrics:**\n"
                        f"  - RMSE: {arima_rmse:.2f}\n"
                        f"  - MAPE: {arima_mape:.2f}\n"
                    )

                    with st.expander("📊 ARIMA Model Summary"):
                        st.markdown(summary_text)

                    # Plot ARIMA Forecast
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], mode="lines", name="Historical", line=dict(color="black", width=2)))
                    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines", name="Forecast", line=dict(color="green", width=2)))

                    fig.update_layout(
                        title="📈 ARIMA Forecast",
                        xaxis_title="Date",
                        yaxis_title="Sales",
                        legend_title="Legend",
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Populate results dictionary
                    results["ARIMA"] = {
                        "RMSE": float(arima_rmse),
                        "MAPE": float(arima_mape),
                        "Forecast": forecast_df
                    }

                except Exception as e:
                    st.warning(f"❌ ARIMA Model failed: {e}")

                # XGBoost Model with Dynamic Adaptation
                st.write("Training XGBoost Model...")

                try:
                    # Step 1: Dynamic Feature Engineering
                    max_lag = min(12, len(train) - 1)  # Adapt lags based on dataset size
                    rolling_windows = [3, 6] if len(train) > 6 else [3]  # Use multiple rolling windows if data permits

                    xgb_data = train.copy()

                    # Create lag features dynamically
                    for lag in range(1, max_lag + 1):
                        xgb_data[f"lag_{lag}"] = xgb_data["y"].shift(lag)

                    # Add rolling statistics dynamically
                    for window in rolling_windows:
                        xgb_data[f"rolling_mean_{window}"] = xgb_data["y"].rolling(window=window).mean()
                        xgb_data[f"rolling_std_{window}"] = xgb_data["y"].rolling(window=window).std()

                    # Add time-based features
                    xgb_data["month"] = xgb_data["ds"].dt.month
                    xgb_data["quarter"] = xgb_data["ds"].dt.quarter
                    xgb_data["year"] = xgb_data["ds"].dt.year
                    xgb_data["sin_month"] = np.sin(2 * np.pi * xgb_data["month"] / 12)
                    xgb_data["cos_month"] = np.cos(2 * np.pi * xgb_data["month"] / 12)

                    # Drop missing values after feature engineering
                    xgb_data.dropna(inplace=True)

                    # Prepare training data
                    feature_cols = [col for col in xgb_data.columns if col not in ["y", "ds"]]
                    x_train = xgb_data[feature_cols]
                    y_train = xgb_data["y"]

                    # Step 2: Train XGBoost Model
                    final_model = XGBRegressor(
                        n_estimators=50,  # Adjust dynamically if needed
                        max_depth=min(5, max(2, len(train) // 10)),  # Adapt max depth to dataset size
                        learning_rate=0.1 if len(train) > 50 else 0.2,  # Adjust LR for larger datasets
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1
                    )
                    final_model.fit(x_train, y_train)

                    # Step 3: Forecast Future Values
                    future_features = []
                    for i in range(forecast_period):
                        future_row = {}

                        # Generate lag features dynamically
                        for lag in range(1, max_lag + 1):
                            future_row[f"lag_{lag}"] = train["y"].iloc[-lag] if lag <= len(train) else np.nan

                        # Generate rolling features dynamically
                        for window in rolling_windows:
                            future_row[f"rolling_mean_{window}"] = (
                                train["y"].rolling(window=min(window, len(train))).mean().iloc[-1]
                                if len(train) > 1
                                else np.nan
                            )
                            future_row[f"rolling_std_{window}"] = (
                                train["y"].rolling(window=min(window, len(train))).std().iloc[-1]
                                if len(train) > 1
                                else np.nan
                            )

                        # Generate seasonal features dynamically
                        future_row["month"] = (train["ds"].iloc[-1] + pd.DateOffset(months=i + 1)).month
                        future_row["quarter"] = (train["ds"].iloc[-1] + pd.DateOffset(months=i + 1)).quarter
                        future_row["year"] = (train["ds"].iloc[-1] + pd.DateOffset(months=i + 1)).year
                        future_row["sin_month"] = np.sin(2 * np.pi * future_row["month"] / 12)
                        future_row["cos_month"] = np.cos(2 * np.pi * future_row["month"] / 12)

                        future_features.append(future_row)

                    # Convert to DataFrame and forward-fill missing values
                    future_df = pd.DataFrame(future_features)
                    future_df.fillna(method="ffill", inplace=True)

                    # Predict future values
                    xgb_forecast = final_model.predict(future_df)

                    # Ensure test['y'] length matches xgb_forecast
                    matching_length = min(len(test["y"]), len(xgb_forecast))
                    test_y_trimmed = test["y"].iloc[:matching_length]
                    xgb_forecast_trimmed = xgb_forecast[:matching_length]

                    # Calculate RMSE and MAPE with matched lengths
                    rmse = mean_squared_error(test_y_trimmed, xgb_forecast_trimmed) ** 0.5  # RMSE = sqrt(MSE)
                    mape = mean_absolute_percentage_error(test_y_trimmed, xgb_forecast_trimmed)

                    # Save Results
                    forecast_df = pd.DataFrame({
                        "ds": pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1), periods=forecast_period, freq="M"),
                        "yhat": xgb_forecast
                    })

                    highest_point = forecast_df.loc[forecast_df["yhat"].idxmax()]
                    lowest_point = forecast_df.loc[forecast_df["yhat"].idxmin()]

                    summary_text = (
                        f"### Key Insights\n"
                        f"- **Projected Growth:** Sales are expected to {'increase' if xgb_forecast[-1] > test['y'].iloc[-1] else 'decrease'} "
                        f"by {abs((xgb_forecast[-1] - test['y'].iloc[-1]) / test['y'].iloc[-1]) * 100:.2f}% in the next period.\n"
                        f"- **Highest Predicted Sales:** {highest_point['yhat']:.2f} on {highest_point['ds'].strftime('%Y-%m-%d')}\n"
                        f"- **Lowest Predicted Sales:** {lowest_point['yhat']:.2f} on {lowest_point['ds'].strftime('%Y-%m-%d')}\n"
                        f"- **Performance Metrics:**\n"
                        f"  - RMSE: {rmse:.2f}\n"
                        f"  - MAPE: {mape:.2f}\n"
                    )

                    with st.expander("📊 XGBoost Model Summary"):
                        st.markdown(summary_text)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], mode="lines", name="Historical", line=dict(color="black", width=2)))
                    fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines", name="Forecast", line=dict(color="red", width=2)))

                    fig.update_layout(
                        title="XGBoost Forecast",
                        xaxis_title="Date",
                        yaxis_title="Sales",
                        legend_title="Legend",
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Populate results dictionary
                    results["XGBoost"] = {
                        "RMSE": float(rmse),
                        "MAPE": float(mape),
                        "Forecast": forecast_df
                    }

                except Exception as e:
                    st.warning(f"XGBoost Model failed: {e}")

                # AutoML Model
                st.write("🚀 Training AutoML Model...")

                try:
                    # ✅ Step 1: Feature Engineering
                    automl_data = train.copy()

                    # ✅ Dynamically determine max_lag based on dataset size
                    if len(train) <= 6:
                        max_lag = min(3, len(train) - 1)
                    elif len(train) <= 12:
                        max_lag = min(6, len(train) - 1)
                    elif len(train) <= 24:
                        max_lag = min(12, len(train) - 1)
                    else:
                        max_lag = min(24, len(train) - 1)

                    st.write(f"🔹 Adjusted Max Lags Used: {max_lag}")

                    # ✅ Add lag features
                    for lag in range(1, max_lag + 1):
                        automl_data[f"lag_{lag}"] = automl_data["y"].shift(lag)

                    # ✅ Add rolling statistics
                    for window in [3, 6, 12]:
                        automl_data[f"rolling_mean_{window}"] = automl_data["y"].rolling(window=window, min_periods=1).mean()
                        automl_data[f"rolling_std_{window}"] = automl_data["y"].rolling(window=window, min_periods=1).std()

                    # ✅ Add trend indicator (YoY growth)
                    if len(train) > 12:
                        automl_data["yoy_growth"] = (automl_data["y"] / automl_data["y"].shift(12)) - 1
                    else:
                        automl_data["yoy_growth"] = 0

                    # ✅ Add momentum tracking
                    automl_data["y_diff"] = automl_data["y"].diff().fillna(0)
                    automl_data["rolling_mean_growth"] = automl_data["y"].rolling(window=3).mean().diff().fillna(0)

                    # ✅ Add seasonal features
                    automl_data["sin_month"] = np.sin(2 * np.pi * automl_data["ds"].dt.month / 12)
                    automl_data["cos_month"] = np.cos(2 * np.pi * automl_data["ds"].dt.month / 12)

                    # ✅ Apply log transformation only when needed
                    if automl_data["y"].max() / automl_data["y"].min() > 5:
                        st.write("🔹 Applying log transformation for variance stabilization.")
                        automl_data["y_log"] = np.log1p(automl_data["y"])
                        apply_log = True
                    else:
                        st.write("🔹 Skipping log transformation.")
                        apply_log = False

                    # Use y instead of y_log if log transformation is skipped
                    if not apply_log:
                        automl_data["y_log"] = automl_data["y"]  # Ensure y_log exists for consistency

                    # Debugging: Check columns
                    st.write(f"🔹 Log transformation applied: {apply_log}")
                    st.write(f"🔹 Columns in automl_data: {automl_data.columns}")

                    automl_data.dropna(inplace=True)

                    # ✅ Dynamically build feature list
                    feature_cols = [col for col in automl_data.columns if col not in ["y", "ds", "y_log"]]

                    # ✅ Prepare training data
                    if apply_log:
                        y_train = automl_data["y_log"]
                    else:
                        y_train = automl_data["y"]  # Use y if log transformation is skipped

                    x_train = automl_data[feature_cols]

                    # ✅ Train AutoML with multiple models
                    automl_model = AutoML()
                    st.write("🔄 **Training AutoML Model...**")
                    automl_model.fit(
                        X_train=x_train,
                        y_train=y_train,
                        task="regression",
                        time_budget=300,
                        eval_method="cv",
                        estimator_list=["xgboost", "lgbm", "rf", "catboost"],
                        metric="r2",
                    )

                    st.write(f"✅ AutoML Training Completed! Best Estimator: {automl_model.best_estimator}")

                    # ✅ Generate Future Data
                    future_features = []
                    last_row = automl_data.iloc[-1].copy()

                    # Debugging: Check columns in last_row
                    st.write(f"🔹 Columns in last_row: {last_row.index.tolist()}")

                    for i in range(forecast_period):
                        future_row = {}

                        # Update lags
                        for lag in range(1, max_lag + 1):
                            if lag == 1:
                                if apply_log:
                                    future_row[f"lag_{lag}"] = last_row["y_log"]  # Use y_log if log transformation is applied
                                else:
                                    future_row[f"lag_{lag}"] = last_row["y"]  # Use y if log transformation is skipped
                            else:
                                future_row[f"lag_{lag}"] = last_row[f"lag_{lag - 1}"]

                        # Update rolling statistics
                        for window in [3, 6, 12]:
                            if apply_log:
                                future_row[f"rolling_mean_{window}"] = last_row[f"rolling_mean_{window}"] + (last_row["y_log"] - last_row[f"lag_{window}"]) / window
                            else:
                                future_row[f"rolling_mean_{window}"] = last_row[f"rolling_mean_{window}"] + (last_row["y"] - last_row[f"lag_{window}"]) / window
                            future_row[f"rolling_std_{window}"] = last_row[f"rolling_std_{window}"]

                        # Update other features
                        future_row["yoy_growth"] = last_row["yoy_growth"]
                        future_row["y_diff"] = last_row["y_diff"]
                        future_row["rolling_mean_growth"] = last_row["rolling_mean_growth"]
                        future_row["sin_month"] = np.sin(2 * np.pi * (last_row["ds"].month + i) / 12)
                        future_row["cos_month"] = np.cos(2 * np.pi * (last_row["ds"].month + i) / 12)

                        # Append the future_row to future_features
                        future_features.append(future_row)

                        # Update last_row for the next iteration
                        last_row = last_row.copy()  # Create a copy of last_row to avoid modifying the original
                        for key, value in future_row.items():
                            last_row[key] = value  # Update last_row with the new values

                    # Convert future_features to a DataFrame
                    future_df = pd.DataFrame(future_features)

                    # ✅ Ensure future data matches training data
                    for col in x_train.columns:
                        if col not in future_df.columns:
                            future_df[col] = 0

                    future_df = future_df[x_train.columns]

                    # ✅ Generate Forecast
                    automl_forecast = automl_model.predict(future_df)

                    # ✅ Reverse log transformation if applied
                    if apply_log:
                        automl_forecast = np.expm1(automl_forecast)

                    # ✅ Prepare Forecast DataFrame
                    forecast_df = pd.DataFrame({
                        "ds": pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1), periods=forecast_period, freq="M"),
                        "yhat": automl_forecast,
                        "yhat_lower": automl_forecast * 0.9,
                        "yhat_upper": automl_forecast * 1.1
                    })

                    # ✅ Calculate RMSE and MAPE
                    test_y = test["y"].values
                    automl_rmse = np.sqrt(mean_squared_error(test_y, forecast_df["yhat"][:len(test_y)]))
                    automl_mape = mean_absolute_percentage_error(test_y, forecast_df["yhat"][:len(test_y)])

                    # Populate results dictionary
                    results["AutoML"] = {
                        "RMSE": float(automl_rmse),
                        "MAPE": float(automl_mape),
                        "Forecast": forecast_df
                    }

                    st.success("✅ AutoML Forecast Generated Successfully!")

                except Exception as e:
                    st.error(f"❌ AutoML Model failed: {e}")

                # ✅ Ensure at least one model generated forecasts
                if not results:
                    st.error("⚠️ No models successfully generated forecasts. Please check your input data.")
                    st.stop()

                # 📊 Model Performance Table
                st.subheader("📌 Model Performance Comparison")

                # Create a DataFrame for model comparison
                comparison_data = []
                for model, result in results.items():
                    if isinstance(result, dict) and "RMSE" in result and "MAPE" in result:
                        try:
                            rmse = float(result["RMSE"])
                            mape = float(result["MAPE"])
                            comparison_data.append({
                                "Model": model,
                                "RMSE": rmse,
                                "MAPE": mape
                            })
                        except (TypeError, ValueError) as e:
                            st.warning(f"⚠️ Invalid RMSE or MAPE for model {model}: {e}")
                    else:
                        st.warning(f"⚠️ Invalid result format for model {model}. Expected a dictionary with 'RMSE' and 'MAPE' keys.")

                if comparison_data:
                    comparison = pd.DataFrame(comparison_data)
                    comparison = comparison.sort_values(by="RMSE")  # Sort models by accuracy (lower RMSE is better)
                    st.dataframe(comparison.style.highlight_min(subset=["RMSE", "MAPE"], color="lightgreen"))

                    # 🏆 AI-Selected Best Model
                    best_model = comparison.iloc[0]["Model"]
                    st.success(f"✨ **AI-Selected Best Model:** {best_model}")

                    # ✅ Validate forecast data
                    if best_model and best_model in results and "Forecast" in results[best_model] and isinstance(results[best_model]["Forecast"], pd.DataFrame) and not results[best_model]["Forecast"].empty:
                        forecast_data = results[best_model]["Forecast"]
                    else:
                        st.warning("⚠️ No valid forecast data available.")
                        forecast_data = None
                else:
                    st.error("⚠️ No valid model results available for comparison.")
                    best_model = None
                    forecast_data = None

                # 🔥 Detect High-Risk Periods in Forecast
                if forecast_data is not None:
                    try:
                        forecast_data["volatility"] = forecast_data["yhat"].rolling(3).std()

                        # Define risk levels dynamically
                        forecast_data["risk"] = "✅ Stable"
                        forecast_data.loc[forecast_data["volatility"] > forecast_data["volatility"].quantile(0.75), "risk"] = "⚠️ High Volatility"
                        forecast_data.loc[forecast_data["volatility"] > forecast_data["volatility"].quantile(0.9), "risk"] = "❌ Major Decline"

                        st.markdown("### 🚨 High-Risk Sales Periods Identified")
                        st.dataframe(forecast_data[["ds", "yhat", "volatility", "risk"]].style.applymap(
                            lambda x: "background-color: #FFDDC1" if x == "❌ Major Decline" else
                                    "background-color: #FFEEAA" if x == "⚠️ High Volatility" else
                                    "background-color: #C6ECAE",
                            subset=["risk"]
                        ))

                        # 📊 AI-Powered Insights
                        highest_point = forecast_data.loc[forecast_data["yhat"].idxmax()]
                        lowest_point = forecast_data.loc[forecast_data["yhat"].idxmin()]
                        projected_growth = ((forecast_data["yhat"].iloc[-1] - test["y"].iloc[-1]) / test["y"].iloc[-1]) * 100
                        trend = "📈 **Growth Expected**" if projected_growth > 0 else "📉 **Potential Decline**"

                        insights_text = f"""
                        - **Projected Sales Growth:** {abs(projected_growth):.2f}% {trend}
                        - **Peak Sales Expected:** ${highest_point['yhat']:.2f} on {highest_point['ds'].strftime('%Y-%m-%d')}
                        - **Lowest Predicted Sales:** ${lowest_point['yhat']:.2f} on {lowest_point['ds'].strftime('%Y-%m-%d')}
                        - **Optimal Decision Window:** Plan around peak sales in {highest_point['ds'].strftime('%B %Y')}
                        - **Risk Zones Identified:** Check months marked as 🔥 'High-Risk' above
                        - **Volatility Analysis:** Forecast suggests a {'stable' if abs(projected_growth) < 5 else 'fluctuating'} trend
                        """

                        with st.expander("🔮 AI-Powered Future Insights"):
                            st.markdown(insights_text)
                    except Exception as e:
                        st.error(f"❌ Error analyzing forecast data: {e}")

                # 📊 Multi-Model Forecast Visualization
                st.markdown("### 🔍 Forecast Comparison Across Models")
                model_colors = {
                    "Prophet": "blue",
                    "ARIMA": "green",
                    "XGBoost": "red",
                    "AutoML": "purple"
                }

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=train["ds"],
                    y=train["y"],
                    mode="lines",
                    name="Historical Data",
                    line=dict(color="black", width=2)
                ))

                for model, result in results.items():
                    if "Forecast" in result and result["Forecast"] is not None and not result["Forecast"].empty:
                        forecast_df = result["Forecast"]
                        fig.add_trace(go.Scatter(
                            x=forecast_df["ds"],
                            y=forecast_df["yhat"],
                            mode="lines",
                            name=f"{model} Forecast",
                            line=dict(width=2, color=model_colors.get(model, "gray"))
                        ))

                fig.update_layout(
                    title="📊 Multi-Model Sales Forecast",
                    xaxis_title="Date",
                    yaxis_title="Sales",
                    legend_title="Models",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

                # 📥 Download Forecast Data
                st.markdown("### 📥 Download Forecast Data")
                if forecast_data is not None:
                    try:
                        csv = forecast_data.to_csv(index=False)
                        st.download_button(
                            label="📩 Download Best Model Forecast (CSV)",
                            data=csv,
                            file_name="forecast.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"❌ Error generating download file: {e}")
                else:
                    st.warning("⚠️ No forecast data available for download.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()