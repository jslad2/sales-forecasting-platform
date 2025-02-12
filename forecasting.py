import os

# Force install dependencies in correct order
os.system("pip install --no-cache-dir setuptools wheel numpy==1.21.6 scipy==1.10.1 scikit-learn==1.3.0")

import streamlit as st
from forecasting_tool import main

# Enable Wide Mode
st.set_page_config(layout="wide")

# Apply SynovaAI Website Styles
def apply_synova_styles():
    """
    Apply custom SynovaAI styles to the Streamlit app.
    """
    st.markdown(
        """
        <style>
            /* Full-width layout */
            .block-container {
                max-width: 100%;
                padding: 2rem;
            }

            /* Center Titles */
            h1, h2, h3 {
                text-align: center;
                color: #2B3A42; /* Matching website text color */
                font-family: 'Roboto', sans-serif;
            }

            /* Buttons */
            .stButton button {
                background-color: #64D8CB !important; /* SynovaAI button color */
                color: white !important;
                font-size: 1rem;
                font-weight: bold;
                border-radius: 8px;
                padding: 12px 20px;
                transition: background-color 0.3s ease, transform 0.3s ease;
            }

            /* Button Hover */
            .stButton button:hover {
                background-color: #56BBAF !important; /* Slightly darker hover */
                transform: scale(1.05);
            }

            /* File Uploader Box */
            div.stFileUploader {
                border: 2px dashed #64D8CB; /* SynovaAI themed border */
                padding: 15px;
                background-color: #F8F9FA; /* Light background */
                text-align: center;
                font-family: 'Roboto', sans-serif;
                color: #333333;
            }

            /* Dashboard Background */
            body {
                background-color: #F8F9FA; /* Matching website background */
                font-family: 'Roboto', sans-serif;
                color: #333333;
            }

            /* Link Styling */
            a {
                color: #64D8CB; /* SynovaAI link color */
                text-decoration: underline;
            }

            a:hover {
                color: #56BBAF;
                text-decoration: none;
            }

            /* Subsection Spacing */
            .stMarkdown {
                margin-bottom: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Display Header Section
def display_header():
    """
    Display the header section of the app.
    """
    st.markdown(
        """
        <header style="background-color: #2B3A42; padding: 20px; text-align: center; color: white; border-radius: 12px;">
            <h1 style="margin: 0; font-size: 2.5rem;">SYNOVAAI</h1>
            <p style="margin: 0; font-size: 1.2rem;">Empowering Your Business with Predictive AI</p>
        </header>
        """,
        unsafe_allow_html=True,
    )
    st.write("")  # Add spacing

# Display Introductory Section
def display_intro():
    """
    Display the introductory section of the app.
    """
    st.markdown(
        """
        <section style="text-align: center; padding: 20px 0; margin-top: 20px;">
            <h2 style="color: #2B3A42; font-size: 2rem; font-weight: bold;">Welcome to the Sales Forecasting Dashboard</h2>
            <p style="font-size: 1.1rem; color: #666666; max-width: 800px; margin: 0 auto;">
                Upload your sales data and leverage our AI-driven predictive analytics to generate accurate, actionable sales forecasts.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

# Display File Upload Section
def display_file_upload():
    """
    Display the file upload section of the app.
    """
    st.markdown(
        """
        <div style="margin: 20px 0; text-align: center;">
            <h3 style="color: #2B3A42; font-size: 1.5rem;">Upload Your Sales Data</h3>
            <p style="color: #666666;">Supported file type: CSV</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main Function to Run the Application
def run_app():
    """
    Run the Streamlit app.
    """
    try:
        # Apply custom SynovaAI styles
        apply_synova_styles()

        # Display header
        display_header()

        # Display introductory section
        display_intro()

        # Display file upload section
        display_file_upload()

        # Run the forecasting tool
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Entry Point
if __name__ == "__main__":
    run_app()
