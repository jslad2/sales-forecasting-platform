import os
import uuid
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def get_supabase_client():
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # For debugging purposes only; remove sensitive prints in production!
    print("SUPABASE_URL:", SUPABASE_URL)
    print("SUPABASE_KEY:", SUPABASE_KEY)
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise Exception("Supabase URL or KEY is missing in environment variables.")
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_forecast(user_id, forecast_name, file_path, model_used, time_horizon, forecast_metrics):
    supabase = get_supabase_client()  # Initialize only when function called
    
    file_id = str(uuid.uuid4())
    file_name = f"{user_id}/{file_id}.csv"
    
    # Upload file
    try:
        with open(file_path, "rb") as f:
            upload_response = supabase.storage.from_("forecasts").upload(file_name, f, {"content-type": "text/csv"})
            print("Upload response:", upload_response)
    except Exception as e:
        raise Exception(f"File upload failed: {e}")
    
    # Get public URL of uploaded file
    file_url_response = supabase.storage.from_("forecasts").get_public_url(file_name)
    # Depending on the library version, file_url_response might be an object or dictionary:
    try:
        file_url = file_url_response.publicURL  # if the response attribute is publicURL
    except AttributeError:
        file_url = file_url_response.get("publicURL", None)
    
    if not file_url:
        raise Exception("Failed to get public URL for the uploaded file.")
    
    forecast_data = {
        "user_id": user_id,
        "forecast_name": forecast_name,
        "model_used": model_used,
        "time_horizon": time_horizon,
        "forecast_metrics": forecast_metrics,
        "forecast_file_url": file_url,
        "is_public": False,
    }
    
    # Insert forecast record into Supabase table
    response = supabase.table("forecasts").insert(forecast_data).execute()
    
    if response.error:
        raise Exception(f"Error inserting forecast data: {response.error}")
    
    print("Forecast data insert response:", response.data)
    return response

# Example usage
if __name__ == "__main__":
    # Replace with actual values for testing
    try:
        response = upload_forecast(
            user_id="user123",
            forecast_name="Sample Forecast",
            file_path="path/to/forecast.csv",
            model_used="AutoML",
            time_horizon="3 months",
            forecast_metrics={"RMSE": 10.5, "MAPE": 15.2}
        )
        print("Upload and database insertion successful:", response)
    except Exception as e:
        print(e)
