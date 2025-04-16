import os
import json
import base64
import logging
from flask import Flask, render_template
from dotenv import load_dotenv
from supabase import create_client
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import jwt as pyjwt

# ─── Load Config & Logging ─────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Shared Config ─────────────────────────────────────────────────────────────
SUPABASE_URL           = os.getenv("SUPABASE_URL")
SUPABASE_KEY           = os.getenv("SUPABASE_KEY")
SERVICE_ACCOUNT_BASE64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
SECRET_KEY             = os.getenv("FLASK_SECRET_KEY") or os.urandom(32).hex()

# ─── Init Supabase ──────────────────────────────────────────────────────────────
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("Missing Supabase config")
    raise RuntimeError("Supabase configuration missing.")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── Init GCP Creds ─────────────────────────────────────────────────────────────
credentials = None
if SERVICE_ACCOUNT_BASE64:
    try:
        info = json.loads(base64.b64decode(SERVICE_ACCOUNT_BASE64))
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        logger.info("GCP service account credentials loaded.")
    except Exception as e:
        logger.error(f"GCP credentials load error: {e}")
else:
    logger.warning("No GCP service account provided; some features may be disabled.")

# ─── Create Flask App ───────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(HERE, "templates"),
    static_folder=os.path.join(HERE, "static")
)
app.config["SECRET_KEY"] = SECRET_KEY

# ─── Register Blueprints ────────────────────────────────────────────────────────
from routes.pages import pages_bp
from routes.auth import auth_bp

app.register_blueprint(pages_bp)
app.register_blueprint(auth_bp, url_prefix="/api")

# ─── Error Handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000))
    )
