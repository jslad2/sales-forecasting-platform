import os, json, base64
import datetime
import requests
import logging
from supabase import create_client
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import jwt as pyjwt
from werkzeug.exceptions import Unauthorized
from dotenv import load_dotenv

# ─── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()
SUPABASE_URL           = os.getenv("SUPABASE_URL")
SUPABASE_KEY           = os.getenv("SUPABASE_KEY")
SERVICE_ACCOUNT_BASE64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
SECRET_KEY             = os.getenv("FLASK_SECRET_KEY") or os.urandom(32).hex()
PROJECT_ID             = os.getenv("GCP_PROJECT_ID", "")

# ─── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Supabase Client ────────────────────────────────────────────────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── GCP Credentials ────────────────────────────────────────────────────────────
credentials = None
if SERVICE_ACCOUNT_BASE64:
    info = json.loads(base64.b64decode(SERVICE_ACCOUNT_BASE64))
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

def get_oauth_token():
    if not credentials:
        raise RuntimeError("GCP credentials missing")
    credentials.refresh(Request())
    return credentials.token

# ─── JWT Helpers ────────────────────────────────────────────────────────────────
def generate_jwt(email: str) -> str:
    payload = {"email": email, "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)}
    return pyjwt.encode(payload, SECRET_KEY, algorithm="HS256")

def decode_jwt(token: str) -> dict:
    try:
        return pyjwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except pyjwt.ExpiredSignatureError:
        raise Unauthorized("Token expired")
    except pyjwt.InvalidTokenError:
        raise Unauthorized("Invalid token")
