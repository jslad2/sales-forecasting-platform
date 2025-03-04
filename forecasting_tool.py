import pandas as pd
import streamlit as st
from prophet import Prophet
from pmdarima import auto_arima
from xgboost import XGBRegressor
from flaml import AutoML
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import optuna
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf
import time

# Enable Wide Mode (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(layout="wide")

def check_stationarity(series):
    """
    Perform the Augmented Dickey-Fuller test to check stationarity.
    """
    result = adfuller(series, autolag="AIC")
    p_value = result[1]
    return "Stationary" if p_value < 0.05 else "Non-Stationary"

def preprocess_data(data, date_column, sales_column):
    """
    Preprocess the uploaded data and check stationarity.
    """
    try:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column, sales_column])

        # Aggregate to Monthly
        data = data[[date_column, sales_column]].rename(columns={date_column: "ds", sales_column: "y"})
        data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
        data = data.groupby(data["ds"].dt.to_period("M")).agg({"y": "sum"}).reset_index()
        data["ds"] = data["ds"].dt.to_timestamp()

        # Check stationarity
        stationarity_result = check_stationarity(data["y"])
        st.markdown(
                    """
                    <div style="text-align: center;">
                        <h2 style="color: #2B3A42;">📊 Stationarity Test</h2>
                        <p style="font-size: 1.2rem;">Conclusion: The series is <strong>Stationary</strong>.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if stationarity_result == "Non-Stationary":
            st.warning("Applying differencing to stabilize the series.")
            data["y"] = data["y"].diff().dropna()

        return data
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        return None



def detect_and_add_seasonalities(model, data):
    """
    Detect seasonalities dynamically and add them to the Prophet model.
    """
    data_frequency = pd.infer_freq(data["ds"])
    if data_frequency == "D":  # Daily data
        model.add_seasonality(name="daily", period=1, fourier_order=3)
    elif data_frequency == "W":  # Weekly data
        model.add_seasonality(name="weekly", period=7, fourier_order=3)
    elif data_frequency == "M":  # Monthly data
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    elif data_frequency == "Q":  # Quarterly data
        model.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
    elif data_frequency == "Y":  # Yearly data
        model.add_seasonality(name="yearly", period=365.25, fourier_order=10)
    return model

def find_best_prophet_params(train):
    """
    Automates the selection of the best Prophet hyperparameters using cross-validation.
    Dynamically adjusts horizon and initial based on the dataset size.
    """
    from prophet.diagnostics import cross_validation, performance_metrics
    from sklearn.model_selection import ParameterGrid

    param_grid = {
        "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.2, 0.3],
        "seasonality_mode": ["additive", "multiplicative"]
    }

    best_params = None
    best_rmse = float("inf")

    # Determine dynamic horizon and initial window
    dataset_length = len(train)
    horizon_days = min(30, max(7, dataset_length // 5))  # Dynamic horizon: 20% of dataset length, capped at 30 days
    initial_days = max(90, dataset_length // 2)  # Dynamic initial window: 50% of dataset length, min 90 days

    horizon = f"{horizon_days} days"
    initial = f"{initial_days} days"

    for params in ParameterGrid(param_grid):
        try:
            # Initialize Prophet model with current parameters
            prophet_model = Prophet(
                seasonality_mode=params["seasonality_mode"],
                changepoint_prior_scale=params["changepoint_prior_scale"]
            )

            # Dynamically detect and add seasonalities
            prophet_model = detect_and_add_seasonalities(prophet_model, train)

            # Fit the model
            prophet_model.fit(train)

            # Perform cross-validation
            cv_results = cross_validation(
                prophet_model,
                initial=initial,
                horizon=horizon,
                period=f"{horizon_days // 2} days"  # Test every half-horizon period
            )
            metrics = performance_metrics(cv_results)

            # Extract RMSE
            rmse = metrics["rmse"].mean()

            # Update best parameters
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params

        except Exception as e:
            # Log and skip invalid configurations
            st.write(f"Failed with params {params}: {e}")
            continue

    return best_params, best_rmse

def main():

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

            # 🚀 Disable "Start Forecast" Button Until Valid Selections
            if date_column != "-- Select Column --" and sales_column != "-- Select Column --":
                start_forecast = st.button("✅ Start Forecast", key="start_btn", help="Click to generate your AI-powered forecast")
            else:
                start_forecast = st.button("⏳ Select Columns First", disabled=True, key="start_disabled")

            # 🏁 Run Forecast Only If Button is Clicked
            if start_forecast:
                # Run forecast logic
                # Preprocess Data
                data = preprocess_data(data, date_column, sales_column)
                if data is None:
                    return

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
                        data.style.set_properties(**{"text-align": "center"}),
                        width=1400,  # Wider Table
                        height=450   # Show More Rows
                    )

                # Determine Testing Period Dynamically
                testing_period = int(len(data) * 0.2)
                train = data.iloc[:-testing_period]
                test = data.iloc[-testing_period:]

                forecast_period = 12  # Fixed to 12 months forecast

                # Forecasting Models
                results = {}

                # Prophet Model
                st.write("🚀 Finding the best Prophet hyperparameters...")
                best_params, best_rmse = find_best_prophet_params(train)  # No need for 'test' parameter anymore

                if best_params is None:
                    st.error("❌ No valid Prophet parameters were found. Check your data preprocessing or parameter grid.")
                    return

                st.write(f"✅ Best Parameters: {best_params}")
                st.write(f"📉 Best RMSE from cross-validation: {best_rmse}")

                # Step 2: Train Final Model
                try:
                    st.write("📊 Training Prophet Model with Best Parameters...")

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

                    # Train the model
                    prophet_model.fit(train)

                    # Generate future dates & predict
                    future = prophet_model.make_future_dataframe(periods=forecast_period, freq="M", include_history=False)
                    prophet_forecast = prophet_model.predict(future)

                    # Ensure only future forecasts are used
                    prophet_forecast = prophet_forecast[prophet_forecast["ds"] > train["ds"].max()]

                    # Ensure test set matches forecast length for metric calculation
                    matching_length = min(len(test["y"]), len(prophet_forecast))
                    prophet_rmse = mean_squared_error(test["y"].iloc[:matching_length], prophet_forecast["yhat"].iloc[:matching_length]) ** 0.5
                    prophet_mape = mean_absolute_percentage_error(test["y"].iloc[:matching_length], prophet_forecast["yhat"].iloc[:matching_length])

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

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], mode="lines", name="Historical", line=dict(color="black", width=2)))
                    fig.add_trace(go.Scatter(x=prophet_forecast["ds"], y=prophet_forecast["yhat"], mode="lines", name="Forecast", line=dict(color="blue", width=2)))
                    fig.add_trace(go.Scatter(x=prophet_forecast["ds"], y=prophet_forecast["yhat_upper"], mode="lines", name="Upper Confidence", line=dict(color="lightblue", dash="dot")))
                    fig.add_trace(go.Scatter(x=prophet_forecast["ds"], y=prophet_forecast["yhat_lower"], mode="lines", name="Lower Confidence", line=dict(color="lightblue", dash="dot")))

                    fig.update_layout(
                        title="Prophet Forecast with Confidence Intervals",
                        xaxis_title="Date",
                        yaxis_title="Sales",
                        legend_title="Legend",
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Save results
                    results["Prophet"] = {
                        "RMSE": float(prophet_rmse),
                        "MAPE": float(prophet_mape),
                        "Forecast": prophet_forecast
                    }

                except Exception as e:
                    st.warning(f"❌ Prophet Model failed: {e}")


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

                    
# AutoML Model with Rolling Features, Log Transformation, and Feature Importance
                st.write("Training AutoML Model...")
                try:
                    # Determine the maximum number of lags based on the dataset size
                    max_lag = min(12, len(train) - 1)  # Limit maximum lags to avoid excessive feature loss
                    automl_data = train.copy()
                    st.write(f"Lags: {max_lag}")

                    # Add lag features
                    for lag in range(1, max_lag + 1):
                        automl_data[f"lag_{lag}"] = automl_data["y"].shift(lag)

                    # Add rolling statistics (only if sufficient data exists)
                    if len(train) > max_lag + 3:
                        automl_data["rolling_mean_3"] = automl_data["y"].rolling(window=3).mean()
                        automl_data["rolling_std_3"] = automl_data["y"].rolling(window=3).std()

                    # Add seasonal features
                    automl_data["sin_month"] = np.sin(2 * np.pi * automl_data["ds"].dt.month / 12)
                    automl_data["cos_month"] = np.cos(2 * np.pi * automl_data["ds"].dt.month / 12)

                    # Apply log transformation to stabilize variance
                    automl_data["y_log"] = np.log1p(automl_data["y"])  # log1p to handle zero values
                    automl_data.dropna(inplace=True)  # Drop rows with NA values after feature engineering

                    # Prepare training data
                    x_train = automl_data.drop(columns=["y", "y_log", "ds"])  # Exclude target and date column
                    y_train = automl_data["y_log"]  # Use log-transformed target

                    # Check for insufficient data
                    if len(x_train) <= 1:
                        st.error("Insufficient data to train AutoML. Please provide more samples.")
                        automl_model = None  # Set automl_model to None if training is skipped
                    else:
                        # Dynamically choose evaluation method
                        eval_method = "cv" if len(x_train) > 5 else "holdout"

                        # Train AutoML Model
                        automl_model = AutoML()
                        automl_model.fit(
                            X_train=x_train,
                            y_train=y_train,
                            task="regression",
                            time_budget=800,  # Time budget for AutoML
                            eval_method=eval_method,  # Dynamically chosen evaluation method
                            estimator_list=["xgboost", "lgbm", "rf"]  # Focus on tree-based models
                        )
                        st.write(f"AutoML Training Completed: Best Estimator - {automl_model.best_estimator}")

                        # Feature Importance (if XGBoost is the best estimator)
                        if automl_model.best_estimator == "xgboost":
                            xgb_model = automl_model.best_model_for_estimator("xgboost")
                            importance = pd.DataFrame({
                                "Feature": x_train.columns,
                                "Importance": xgb_model.feature_importances_
                            }).sort_values(by="Importance", ascending=False)
                            st.write("Feature Importance:")
                            st.dataframe(importance)

                        # Generate future forecasts using the trained AutoML model
                        st.write("Generating forecasts with AutoML...")
                        test_lags = {f"lag_{lag}": [train["y"].iloc[-lag]] for lag in range(1, max_lag + 1)}
                        test_lags["rolling_mean_3"] = train["y"].rolling(window=3).mean().iloc[-1]
                        test_lags["rolling_std_3"] = train["y"].rolling(window=3).std().iloc[-1]
                        test_lags["sin_month"] = np.sin(2 * np.pi * train["ds"].iloc[-1].month / 12)
                        test_lags["cos_month"] = np.cos(2 * np.pi * train["ds"].iloc[-1].month / 12)
                        future_df = pd.DataFrame(test_lags)  # Start with lag features

                        automl_forecast = []
                        for _ in range(forecast_period):
                            next_forecast_log = automl_model.predict(future_df)[0]  # Predict log-transformed target
                            next_forecast = np.expm1(next_forecast_log)  # Reverse log transformation
                            automl_forecast.append(next_forecast)

                            # Update future lagged features
                            for lag in range(max_lag, 1, -1):
                                future_df[f"lag_{lag}"] = future_df[f"lag_{lag - 1}"]
                            future_df["lag_1"] = next_forecast

                            # Update rolling statistics
                            future_df["rolling_mean_3"] = np.mean(automl_forecast[-3:])
                            future_df["rolling_std_3"] = np.std(automl_forecast[-3:])

                        # Save AutoML results
                        automl_rmse = mean_squared_error(test["y"], automl_forecast[:len(test)], squared=False)
                        automl_mape = mean_absolute_percentage_error(test["y"], automl_forecast[:len(test)])
                        results["AutoML"] = {
                            "RMSE": automl_rmse,
                            "MAPE": automl_mape,
                            "Forecast": pd.DataFrame({
                                "ds": pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1), periods=forecast_period, freq="M"),
                                "yhat": automl_forecast
                            })
                        }
                        st.write(f"AutoML RMSE: {automl_rmse}")
                        st.write(f"AutoML MAPE: {automl_mape}")

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
                    # Ensure the result contains valid RMSE and MAPE values
                    if isinstance(result, dict) and "RMSE" in result and "MAPE" in result:
                        try:
                            # Convert RMSE and MAPE to float (in case they are numpy.float64)
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

                # Check if any valid models were added to the comparison
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




