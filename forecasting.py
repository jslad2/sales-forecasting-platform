import streamlit as st
from forecasting_tool import main  # Keep forecasting logic separate
import pandas as pd
import plotly.express as px
import datetime

# ✅ Apply custom SynovaAI styles
def apply_styles():
    st.markdown(
        """
        <style>
            body { background-color: #F8F9FA; font-family: 'Roboto', sans-serif; color: #333333; }
            .block-container { max-width: 100%; padding: 2rem; }
            h1, h2, h3 { text-align: center; font-family: 'Roboto', sans-serif; }

            /* Top Banner */
            .banner {
                background: linear-gradient(135deg, #2B3A42, #64D8CB);
                padding: 50px;
                text-align: center;
                border-radius: 12px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                color: white;
                font-size: 2rem;
                font-weight: bold;
            }

            /* Main Title */
            .main-title {
                font-size: 2.5rem;
                color: #2B3A42;
                font-weight: bold;
                text-align: center;
                margin-top: 30px;
            }

            /* Subtext */
            .subtext {
                font-size: 1.2rem;
                color: #444;
                max-width: 800px;
                margin: 0 auto;
                line-height: 1.6;
                text-align: center;
            }

            /* File Upload Box */
            div.stFileUploader {
                border: 2px dashed #64D8CB;
                padding: 15px;
                background-color: #F8F9FA;
                text-align: center;
                font-family: 'Roboto', sans-serif;
                color: #333333;
            }

            /* Button */
            .stButton button {
                background-color: #64D8CB !important;
                color: white !important;
                font-size: 1rem;
                font-weight: bold;
                border-radius: 8px;
                padding: 12px 20px;
                transition: background-color 0.3s ease, transform 0.3s ease;
            }

            .stButton button:hover {
                background-color: #56BBAF !important;
                transform: scale(1.05);
            }

            /* Premium Features Section */
            .premium-feature {
                background: #FFFFFF;
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }

            .premium-feature h3 {
                color: #2B3A42;
                font-size: 1.5rem;
                margin-bottom: 10px;
            }

            .premium-feature p {
                color: #444;
                font-size: 1rem;
                line-height: 1.6;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ✅ Header Section
def display_header():
    st.markdown('<div class="banner">AI-Powered Sales & Demand Forecasting</div>', unsafe_allow_html=True)

# ✅ Introduction Section
def display_intro():
    st.markdown('<h1 class="main-title">Predict the Future with AI-Driven Forecasting</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtext">'
        'SynovaAI helps businesses forecast sales, demand, and inventory with AI-powered predictive analytics. '
        'Gain <strong>data-driven insights</strong> and optimize your strategy today.'
        '</p>',
        unsafe_allow_html=True
    )

# ✅ File Upload Section
def display_file_upload():
    st.markdown(
        """
        <div style="margin: 20px 0; text-align: center;">
            <h3 style="color: #2B3A42; font-size: 1.5rem;">Upload Your Data & Get AI Predictions</h3>
            <p style="color: #444;">Supported file type: <strong>CSV</strong>. Ensure your data includes <strong>Date</strong> and <strong>Sales</strong> columns.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    return uploaded_file

# ✅ Display Model Summary Section
def display_model_summary(best_model, metrics):
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>🔍 Best Model Summary</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <p class="subtext">
            Based on our analysis, the best model for your dataset is <strong>{best_model}</strong>.
            Here’s a quick overview of its performance:
        </p>
        <ul>
            <li><strong>RMSE:</strong> {metrics["RMSE"]:.2f}</li>
            <li><strong>MAPE:</strong> {metrics["MAPE"]:.2%}</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

# ✅ Display Forecast Visualization
def display_forecast_visualization(forecast_data):
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>📊 Forecast Visualization</h2>", unsafe_allow_html=True)
    fig = px.line(forecast_data, x="Date", y="Sales", title="Sales Forecast Over Time")
    st.plotly_chart(fig, use_container_width=True)

# ✅ Display Premium Features
def display_premium_features():
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>✨ Premium Features</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="premium-feature">
            <h3>Advanced Analytics Dashboard</h3>
            <p>Access an interactive dashboard to explore your data in-depth. Visualize trends, seasonality, and anomalies with advanced charts and graphs.</p>
        </div>
        <div class="premium-feature">
            <h3>Customizable Forecast Period</h3>
            <p>Choose your forecast horizon (e.g., 30 days, 90 days, 1 year) and customize the granularity of predictions to suit your business needs.</p>
        </div>
        <div class="premium-feature">
            <h3>Exportable Reports</h3>
            <p>Download detailed PDF reports with insights, forecasts, and recommendations to share with your team or stakeholders.</p>
        </div>
        <div class="premium-feature">
            <h3>Priority Support</h3>
            <p>Get priority access to our support team for any questions or issues. We’re here to help you succeed!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ✅ Main Function
def run_app():
    try:
        apply_styles()  # Apply custom styles
        display_header()  # Display header
        display_intro()  # Display intro section
        uploaded_file = display_file_upload()  # Display file upload section

        if uploaded_file is not None:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.write("### Preview of Uploaded Data")
            st.write(data.head())

            # Run forecasting logic and retrieve results
            results = main(data)

            if results:
                best_model = min(results, key=lambda x: results[x]["RMSE"])
                best_metrics = results[best_model]
                display_model_summary(best_model, best_metrics)

                # Display forecast visualization
                forecast_data = results[best_model]["forecast"]
                display_forecast_visualization(forecast_data)

        # Display premium features
        display_premium_features()

    except Exception as e:
        st.error(f"An error occurred: {e}")

# ✅ Entry Point
if __name__ == "__main__":
    run_app()