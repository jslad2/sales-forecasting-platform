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

# Enable Wide Mode
st.set_page_config(layout="wide")

# --- Custom Styling for Streamlit ---
st.markdown("""
    <style>
        .stApp {
            background-color: #F4F4F6;
            color: #333333;
        }
        h1, h3 {
            color: #2B3A42;
            text-align: center;
        }
        .stButton>button {
            background-color: #2B3A42;
            color: white;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #56BBAF;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.markdown(
    """
    <header style="background-color: #2B3A42; padding: 20px; text-align: center; color: white; border-radius: 12px;">
        <h1 style="margin: 0; font-size: 2.5rem;">Sales Dashboard</h1>
        <p style="margin: 0; font-size: 1.2rem;">Empowering Your Business with Data-Driven Insights</p>
    </header>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def check_stationarity(series):
    """
    Perform the Augmented Dickey-Fuller test to check stationarity.
    """
    result = adfuller(series, autolag="AIC")
    p_value = result[1]
    return "Stationary" if p_value < 0.05 else "Non-Stationary"

@st.cache_data
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
        st.subheader("Stationarity Test")
        st.write(f"Conclusion: The series is **{stationarity_result}**.")

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
    st.title("Sales Forecasting Platform")

    # File Upload
    uploaded_file = st.file_uploader("Upload your sales data file", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(data)

            # Column Mapping
            st.subheader("Map Your Columns")
            date_column = st.selectbox("Select the Date Column:", data.columns)
            sales_column = st.selectbox("Select the Sales Column:", data.columns)

            if st.button("Start Forecast"):
                # Preprocess Data
                data = preprocess_data(data, date_column, sales_column)
                if data is None:
                    return

                st.write("Preprocessed Monthly Data:")
                st.dataframe(data)

                # Determine Testing Period Dynamically
                testing_period = int(len(data) * 0.2)
                train = data.iloc[:-testing_period]
                test = data.iloc[-testing_period:]

                forecast_period = 12  # Fixed to 12 months forecast

                # Forecasting Models
                results = {}

                # Prophet Model
                st.write("Finding the best Prophet hyperparameters...")
                best_params, best_rmse = find_best_prophet_params(train)  # No need for 'test' parameter anymore
                if best_params is None:
                    st.error("No valid Prophet parameters were found. Check your data preprocessing or parameter grid.")
                    return

                st.write(f"Best Parameters: {best_params}")
                st.write(f"Best RMSE from cross-validation: {best_rmse}")

                # Step 2: Train Final Model
                try:
                    st.write("Training Prophet Model with Best Parameters...")
                    prophet_model = Prophet(
                        seasonality_mode=best_params["seasonality_mode"],
                        changepoint_prior_scale=best_params["changepoint_prior_scale"]
                    )

                    # Dynamically add seasonalities
                    prophet_model = detect_and_add_seasonalities(prophet_model, train)

                    # Fit the model
                    prophet_model.fit(train)

                    # Forecast
                    future = prophet_model.make_future_dataframe(periods=forecast_period, freq="M")
                    prophet_forecast = prophet_model.predict(future)

                    # Step 3: Evaluate Performance
                    prophet_rmse = mean_squared_error(test["y"], prophet_forecast["yhat"][-len(test):], squared=False)
                    prophet_mape = mean_absolute_percentage_error(test["y"], prophet_forecast["yhat"][-len(test):])

                    st.write(f"Final Prophet RMSE: {prophet_rmse}")
                    st.write(f"Final Prophet MAPE: {prophet_mape}")

                    # Save Results
                    results["Prophet"] = {
                        "RMSE": prophet_rmse,
                        "MAPE": prophet_mape,
                        "Forecast": prophet_forecast
                    }

                    # Custom Visualization for Forecast
                    st.write(f"Forecast for {forecast_period} Months Using Prophet")
                    fig = go.Figure()

                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=train["ds"], y=train["y"], mode="lines", name="Historical",
                        line=dict(color="black", width=2)
                    ))

                    # Add forecast data
                    fig.add_trace(go.Scatter(
                        x=prophet_forecast["ds"], y=prophet_forecast["yhat"], mode="lines", name="Forecast",
                        line=dict(color="blue", width=2)
                    ))

                    # Add confidence intervals
                    fig.add_trace(go.Scatter(
                        x=prophet_forecast["ds"], y=prophet_forecast["yhat_upper"], mode="lines",
                        name="Upper Confidence", line=dict(color="lightblue", dash="dot")
                    ))
                    fig.add_trace(go.Scatter(
                        x=prophet_forecast["ds"], y=prophet_forecast["yhat_lower"], mode="lines",
                        name="Lower Confidence", line=dict(color="lightblue", dash="dot")
                    ))

                    # Customize layout
                    fig.update_layout(
                        title="Prophet Forecast with Confidence Intervals",
                        xaxis_title="Date",
                        yaxis_title="Sales",
                        legend_title="Legend",
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.warning(f"Failed to train Prophet model: {e}")

                # ARIMA Model
                st.write("Training ARIMA Model...")
                try:
                    # Analyze seasonality dynamically
                    if len(train) <= 12:
                        st.warning("Insufficient data for seasonal differencing. Using non-seasonal ARIMA.")
                        seasonal = False
                    else:
                        st.write("Analyzing seasonality in the data...")
                        decomposition = seasonal_decompose(train["y"], model="additive", period=12)  # Assuming monthly data
                        seasonality_present = np.any(np.abs(decomposition.seasonal) > 0.01)

                        if seasonality_present:
                            st.write("Seasonality detected. Using seasonal ARIMA.")
                            seasonal = True
                        else:
                            st.write("No significant seasonality detected. Using non-seasonal ARIMA.")
                            seasonal = False

                    # Automatically configure ARIMA model
                    arima_model = auto_arima(
                        train["y"],
                        seasonal=seasonal,
                        m=12 if seasonal else 1,
                        d=1,  # Differencing
                        D=1 if seasonal else 0,  # Seasonal differencing
                        trace=True,
                        suppress_warnings=True,
                        error_action="ignore",
                        stepwise=True
                    )

                    # Generate future forecast
                    arima_forecast = arima_model.predict(n_periods=forecast_period)
                    results["ARIMA"] = {
                        "RMSE": mean_squared_error(test["y"], arima_forecast[:len(test)], squared=False),
                        "MAPE": mean_absolute_percentage_error(test["y"], arima_forecast[:len(test)]),
                        "Forecast": pd.DataFrame({
                            "ds": pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1), periods=forecast_period, freq="M"),
                            "yhat": arima_forecast
                        })
                    }

                except Exception as e:
                    st.warning(f"ARIMA Model failed: {e}")


                # XGBoost Model with Improved Feature Engineering and RFE
                st.write("Training XGBoost Model Dynamically...")
                try:
                    # Step 1: Feature Engineering
                    max_lag = min(12, len(train) - 1)
                    xgb_data = train.copy()

                    # Create lag features
                    for lag in range(1, max_lag + 1):
                        xgb_data[f"lag_{lag}"] = xgb_data["y"].shift(lag)

                    # Add rolling statistics
                    rolling_windows = [3, 6]  # Example rolling windows
                    for window in rolling_windows:
                        xgb_data[f"rolling_mean_{window}"] = xgb_data["y"].rolling(window=window).mean()
                        xgb_data[f"rolling_std_{window}"] = xgb_data["y"].rolling(window=window).std()

                    # Add basic time-based features
                    xgb_data["month"] = xgb_data["ds"].dt.month
                    xgb_data["quarter"] = xgb_data["ds"].dt.quarter
                    xgb_data["year"] = xgb_data["ds"].dt.year

                    xgb_data.dropna(inplace=True)

                    # Prepare training data
                    x_train = xgb_data.drop(columns=["y", "ds"])
                    y_train = xgb_data["y"]

                    # Step 2: Recursive Feature Elimination (RFE)
                    st.write("Performing Recursive Feature Elimination...")
                    base_model = XGBRegressor(objective="reg:squarederror", random_state=42)
                    selector = RFE(estimator=base_model, n_features_to_select=10, step=1)  # Select top 10 features
                    selector = selector.fit(x_train, y_train)
                    selected_features = x_train.columns[selector.support_]

                    st.write("Selected Features:")
                    st.dataframe(selected_features)

                    x_train_selected = x_train[selected_features]

                    # Train Final Model
                    st.write("Training Final XGBoost Model...")
                    final_model = XGBRegressor(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        objective="reg:squarederror",
                        random_state=42
                    )
                    final_model.fit(x_train_selected, y_train)

                    # Feature Importance
                    st.write("XGBoost Feature Importance:")
                    feature_importance = pd.DataFrame({
                        "Feature": selected_features,
                        "Importance": final_model.feature_importances_
                    }).sort_values(by="Importance", ascending=False)
                    st.dataframe(feature_importance)

                    # Step 3: Forecast Future Values
                    future_df = pd.DataFrame({f"lag_{lag}": [train["y"].iloc[-lag]] for lag in range(1, max_lag + 1)})
                    for window in rolling_windows:
                        future_df[f"rolling_mean_{window}"] = train["y"].rolling(window=window).mean().iloc[-1]
                        future_df[f"rolling_std_{window}"] = train["y"].rolling(window=window).std().iloc[-1]
                    future_df["month"] = train["ds"].iloc[-1].month
                    future_df["quarter"] = train["ds"].iloc[-1].quarter
                    future_df["year"] = train["ds"].iloc[-1].year

                    xgb_forecast = []
                    for _ in range(forecast_period):
                        next_forecast = final_model.predict(future_df[selected_features])[0]
                        xgb_forecast.append(next_forecast)
                        for lag in range(max_lag, 1, -1):
                            future_df[f"lag_{lag}"] = future_df[f"lag_{lag - 1}"]
                        future_df["lag_1"] = next_forecast
                        for window in rolling_windows:
                            future_df[f"rolling_mean_{window}"] = np.mean(xgb_forecast[-window:]) if len(xgb_forecast) >= window else np.nan
                            future_df[f"rolling_std_{window}"] = np.std(xgb_forecast[-window:]) if len(xgb_forecast) >= window else np.nan

                    # Save Results
                    results["XGBoost"] = {
                        "RMSE": mean_squared_error(test["y"], xgb_forecast[:len(test)], squared=False),
                        "MAPE": mean_absolute_percentage_error(test["y"], xgb_forecast[:len(test)]),
                        "Forecast": pd.DataFrame({
                            "ds": pd.date_range(start=train["ds"].iloc[-1] + pd.DateOffset(months=1), periods=forecast_period, freq="M"),
                            "yhat": xgb_forecast
                        })
                    }

                    st.write(f"Final XGBoost RMSE: {results['XGBoost']['RMSE']}")
                    st.write(f"Final XGBoost MAPE: {results['XGBoost']['MAPE']}")

                except Exception as e:
                    st.warning(f"XGBoost Model failed: {e}")
                    results["XGBoost"] = {
                        "RMSE": float("inf"),
                        "MAPE": float("inf"),
                        "Forecast": pd.DataFrame(columns=["ds", "yhat"])
                    }

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
                            time_budget=300,  # Time budget for AutoML
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
                    st.warning(f"AutoML Model failed: {e}")

                # Check if AutoML training occurred successfully
                if automl_model is None:
                    st.warning("AutoML was not trained due to insufficient data.")
                else:
                    st.success("AutoML training and forecasting completed successfully!")

                # Model Performance Table
                st.subheader("Model Performance Comparison")
                comparison = pd.DataFrame([
                    {"Model": model, "RMSE": result["RMSE"], "MAPE": result["MAPE"]}
                    for model, result in results.items()
                ])
                st.dataframe(comparison)

                # Select Best Model Based on RMSE
                best_model = comparison.loc[comparison["RMSE"].idxmin(), "Model"]
                st.write(f"Best Model: {best_model}")

                # Ensure forecast data exists for the best model
                if best_model in results and "Forecast" in results[best_model] and not results[best_model]["Forecast"].empty:
                    forecast_data = results[best_model]["Forecast"]
                else:
                    forecast_data = None  # Handle missing forecast case

                # Prepare Data for Comparison Chart
                historical_data = train[["ds", "y"]].rename(columns={"y": "yhat"})
                historical_data["Model"] = "Historical"

                # Define colors for different models
                model_colors = {
                    "Prophet": "blue",
                    "ARIMA": "green",
                    "XGBoost": "red",
                    "AutoML": "purple"
                }

                # Create Plotly Figure
                fig = go.Figure()

                # Add Historical Data
                fig.add_trace(go.Scatter(
                    x=historical_data["ds"],
                    y=historical_data["yhat"],
                    mode="lines",
                    name="Historical",
                    line=dict(color="black", width=2)
                ))

                # Add Forecasts for Each Model (Only 12-month Forecast)
                for model, result in results.items():
                    if "Forecast" in result and result["Forecast"] is not None and not result["Forecast"].empty:
                        forecast_df = result["Forecast"]
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_df["ds"],
                            y=forecast_df["yhat"],
                            mode="lines",
                            name=model,
                            line=dict(width=2, color=model_colors.get(model, "gray"))
                        ))

                # Final Plot Formatting
                fig.update_layout(
                    title="Sales Forecast Comparison Across Models",
                    xaxis_title="Date",
                    yaxis_title="Sales",
                    legend_title="Models",
                    template="plotly_white"
                )

                # Show the Chart
                st.plotly_chart(fig, use_container_width=True)

                # Download Forecast
                st.subheader("Download Forecast")
                if forecast_data is not None:
                    st.download_button("Download Forecast Data (CSV)", forecast_data.to_csv(index=False), "forecast.csv", "text/csv")
                else:
                    st.warning("No forecast data available for download.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
