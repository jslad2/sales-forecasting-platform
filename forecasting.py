import streamlit as st
from forecasting_tool import main  # Keep forecasting logic separate

# ✅ Apply custom SynovaAI styles
def apply_styles():
    st.markdown(
        """
        <style>
            body { background-color: #F8F9FA; font-family: 'Roboto', sans-serif; color: #333333; }
            .block-container { max-width: 100%; padding: 2rem; }
            h1, h2, h3 { text-align: center; color: #2B3A42; font-family: 'Roboto', sans-serif; }
            .stButton button { background-color: #64D8CB !important; color: white !important; font-size: 1rem; font-weight: bold; border-radius: 8px; padding: 12px 20px; }
            .stButton button:hover { background-color: #56BBAF !important; transform: scale(1.05); }
            div.stFileUploader { border: 2px dashed #64D8CB; padding: 15px; background-color: #F8F9FA; text-align: center; font-family: 'Roboto', sans-serif; color: #333333; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ✅ Header Section
def display_header():
    st.markdown(
        """
        <header style="background-color: #2B3A42; padding: 30px; text-align: center; color: white; border-radius: 12px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);">
            <h1 style="margin: 0; font-size: 2.5rem;">SynovaAI Forecasting</h1>
            <p style="margin: 0; font-size: 1.2rem;">AI-Powered Sales & Demand Forecasting</p>
        </header>
        """,
        unsafe_allow_html=True,
    )

# ✅ Introduction Section
def display_intro():
    st.markdown(
        """
        <section style="text-align: center; padding: 30px 0;">
            <h2 style="color: #2B3A42; font-size: 2rem; font-weight: bold;">Predict the Future with AI-Driven Forecasting</h2>
            <p style="font-size: 1.1rem; color: #444; max-width: 800px; margin: 0 auto; line-height: 1.6;">
                SynovaAI helps businesses forecast sales, demand, and inventory with AI-powered predictive analytics.
                Gain <strong>data-driven insights</strong> and optimize your strategy today.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
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

# ✅ Main Function
def run_app():
    try:
        apply_styles()  # Apply custom styles
        display_header()  # Display header
        display_intro()  # Display intro section
        display_file_upload()  # Display file upload section
        main()  # Run the forecasting tool
    except Exception as e:
        st.error(f"An error occurred: {e}")

# ✅ Entry Point
if __name__ == "__main__":
    run_app()
