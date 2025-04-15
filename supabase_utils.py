import os
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime
import uuid

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_forecast(user_id, forecast_name, file_path, model_used, time_horizon, forecast_metrics):
    # Upload CSV to storage
    file_id = str(uuid.uuid4())
    file_name = f"{user_id}/{file_id}.csv"
    
    with open(file_path, "rb") as f:
        supabase.storage.from_("forecasts").upload(file_name, f, {"content-type": "text/csv"})

    # Generate public URL
    file_url = supabase.storage.from_("forecasts").get_public_url(file_name)
    
    # Insert metadata into the DB
    forecast_data = {
        "user_id": user_id,
        "forecast_name": forecast_name,
        "model_used": model_used,
        "time_horizon": time_horizon,
        "forecast_metrics": forecast_metrics,
        "forecast_file_url": file_url,
        "is_public": False,
    }
    
    response = supabase.table("forecasts").insert(forecast_data).execute()
    return response
