import os
import uuid
from datetime import datetime
from supabase import create_client
import streamlit as st

# Optionally load local .env for local development
# In Streamlit Cloud, st.secrets is preferred.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_supabase_client():
    # Try to get the values from st.secrets first.
    SUPABASE_URL = st.secrets.get("SUPABASE_URL") if hasattr(st, "secrets") else None
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") if hasattr(st, "secrets") else None

    # Fallback to environment variables if secrets are not available.
    if not SUPABASE_URL:
        SUPABASE_URL = os.getenv("SUPABASE_URL")
    if not SUPABASE_KEY:
        SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # For debugging purposes only; remove sensitive prints in production.
    print("SUPABASE_URL:", SUPABASE_URL)
    print("SUPABASE_KEY:", SUPABASE_KEY)
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise Exception("Supabase URL or KEY is missing in your configuration. Check your st.secrets or .env file.")
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_forecast(user_id, forecast_name, file_path, model_used, time_horizon, forecast_metrics):
    supabase = get_supabase_client()  # Initialize only when function is called
    
    file_id = str(uuid.uuid4())
    file_name = f"{user_id}/{file_id}.csv"
    
    # Upload file to the "forecasts" bucket
    try:
        with open(file_path, "rb") as f:
            upload_response = supabase.storage.from_("forecasts").upload(
                file_name, f, {"content-type": "text/csv"}
            )
            print("Upload response:", upload_response)
    except Exception as e:
        raise Exception(f"File upload failed: {e}")
    
    # Get public URL of the uploaded file
    file_url_response = supabase.storage.from_("forecasts").get_public_url(file_name)
    
    # Depending on the library version, file_url_response may have a publicURL attribute or be a dict.
    try:
        file_url = file_url_response.publicURL
    except AttributeError:
        file_url = file_url_response.get("publicURL", None)
    
    if not file_url:
        raise Exception("Failed to obtain a public URL for the uploaded file.")
    
    forecast_data = {
        "user_id": user_id,
        "forecast_name": forecast_name,
        "model_used": model_used,
        "time_horizon": time_horizon,
        "forecast_metrics": forecast_metrics,
        "forecast_file_url": file_url,
        "is_public": False,
    }
    
    # Insert forecast record into the Supabase "forecasts" table
    response = supabase.table("forecasts").insert(forecast_data).execute()
    
    if response.error:
        raise Exception(f"Error inserting forecast data: {response.error}")
    
    print("Forecast data insert response:", response.data)
    return response

# Example usage; be sure to update the file_path to an actual file for testing.
if __name__ == "__main__":
    try:
        response = upload_forecast(
            user_id="user123",
            forecast_name="Sample Forecast",
            file_path="path/to/forecast.csv",  # Update this path accordingly.
            model_used="AutoML",
            time_horizon="3 months",
            forecast_metrics={"RMSE": 10.5, "MAPE": 15.2}
        )
        print("Upload and database insertion successful:", response)
    except Exception as e:
        print(e)
