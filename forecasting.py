import streamlit as st
from forecasting_tool import main  # Keep forecasting logic separate

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

# ✅ Main Function
def run_app():
    try:
        apply_styles()  # Apply custom styles
        display_header()  # Display header
        display_intro()  # Display intro section
        display_file_upload()  # Display file upload section
        
        # Run forecasting logic and retrieve results
        results = main()

        if results:
            best_model = min(results, key=lambda x: results[x]["RMSE"])
            best_metrics = results[best_model]
            display_model_summary(best_model, best_metrics)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# ✅ Entry Point
if __name__ == "__main__":
    run_app()
