import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Forecasting Models
from prophet import Prophet
from pmdarima import auto_arima
from xgboost import XGBRegressor
from flaml import AutoML

# Model Evaluation & Feature Engineering
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import RFE

# Stationarity Test
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Enable Wide Mode
st.set_page_config(layout="wide")

# --- Header ---
st.markdown(
    """
    <header style="background-color: #2B3A42; padding: 20px; text-align: center; color: white; border-radius: 12px;">
        <h1 style="margin: 0; font-size: 2.5rem;">Sales Forecasting Dashboard</h1>
        <p style="margin: 0; font-size: 1.2rem;">Get Actionable Insights with AI-Powered Forecasting</p>
    </header>
    """,
    unsafe_allow_html=True,
)

# Prophet Model Implementation
def train_prophet_model(train, test, forecast_period=12):
    """
    Train Prophet model with optimized parameters and return forecast.
    """
    st.write("Finding the best Prophet hyperparameters...")
    best_params, best_rmse = find_best_prophet_params(train)
    if best_params is None:
        st.error("No valid Prophet parameters were found. Check your data preprocessing or parameter grid.")
        return None

    st.write(f"Best Parameters: {best_params}")
    st.write(f"Best RMSE from cross-validation: {best_rmse}")

    try:
        st.write("Training Prophet Model with Best Parameters...")
        prophet_model = Prophet(
            seasonality_mode=best_params["seasonality_mode"],
            changepoint_prior_scale=best_params["changepoint_prior_scale"]
        )

        prophet_model = detect_and_add_seasonalities(prophet_model, train)
        prophet_model.fit(train)

        future = prophet_model.make_future_dataframe(periods=forecast_period, freq="M")
        prophet_forecast = prophet_model.predict(future)

        prophet_rmse = mean_squared_error(test["y"], prophet_forecast["yhat"].iloc[-len(test):]) ** 0.5
        prophet_mape = mean_absolute_percentage_error(test["y"], prophet_forecast["yhat"].iloc[-len(test):])

        summary_text = (
            f"### Key Insights\n"
            f"- **Projected Growth:** Sales are expected to {'increase' if prophet_forecast['yhat'].iloc[-1] > test['y'].iloc[-1] else 'decrease'} by {abs(((prophet_forecast['yhat'].iloc[-1] - test['y'].iloc[-1]) / test['y'].iloc[-1]) * 100):.2f}% in the next period.\n"
            f"- **Best Model Parameters:** {best_params}\n"
            f"- **Performance Metrics:**\n"
            f"  - RMSE: {prophet_rmse:.2f}\n"
            f"  - MAPE: {prophet_mape:.2f}\n"
        )

        results = {
            "RMSE": prophet_rmse,
            "MAPE": prophet_mape,
            "Forecast": prophet_forecast
        }

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

        return results
    except Exception as e:
        st.warning(f"Failed to train Prophet model: {e}")
        return None
