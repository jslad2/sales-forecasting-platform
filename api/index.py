import os
import re
import uuid
import json
import base64
import logging
import datetime
import requests
import concurrent.futures
from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, flash, send_from_directory, abort, g
)
from werkzeug.exceptions import Unauthorized, BadRequest
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client, Client
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, ReplyTo
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import jwt as pyjwt  # avoid name collision

# ─── Load Config ────────────────────────────────────────────────────────────────
load_dotenv()

# Streamlit-secret style would go here for Streamlit; for Flask we rely on .env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_SENDER = os.getenv("SENDGRID_SENDER")
RECAPTCHA_SITE_KEY = os.getenv("RECAPTCHA_SITE_KEY")
RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY")
SERVICE_ACCOUNT_BASE64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
SECRET_KEY = os.getenv("FLASK_SECRET_KEY") or os.urandom(32).hex()
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "synovaai-1741395134509")

# ─── Determine the directory this file lives in ────────────────────────────────
HERE = os.path.dirname(__file__)

# ─── Build absolute paths to your local templates/ and static/ folders ─────────
template_dir = os.path.join(HERE, "templates")
static_dir   = os.path.join(HERE, "static")

# ─── Initialize Flask, pointing at those folders ───────────────────────────────
app = Flask(
    __name__,
    template_folder=template_dir,
    static_folder=static_dir
)

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Validate & Init Supabase ───────────────────────────────────────────────────
if not SUPABASE_URL or not SUPABASE_KEY:
    logger.critical("Supabase credentials are missing. Check your .env.")
    raise RuntimeError("Supabase configuration missing.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── Load Google Service Account ────────────────────────────────────────────────
credentials = None
if SERVICE_ACCOUNT_BASE64:
    try:
        info = json.loads(base64.b64decode(SERVICE_ACCOUNT_BASE64))
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        logger.info("Loaded Google service account credentials.")
    except Exception as e:
        logger.error(f"Failed to load GCP credentials: {e}")
else:
    logger.warning("No GCP service account provided.")

def get_oauth_token():
    if not credentials:
        raise RuntimeError("GCP credentials not initialized.")
    credentials.refresh(Request())
    return credentials.token

# ─── JWT Helpers ────────────────────────────────────────────────────────────────
def generate_jwt(email: str) -> str:
    payload = {
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return pyjwt.encode(payload, SECRET_KEY, algorithm="HS256")

def decode_jwt(token: str) -> dict:
    try:
        return pyjwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except pyjwt.ExpiredSignatureError:
        raise Unauthorized("Token expired")
    except pyjwt.InvalidTokenError:
        raise Unauthorized("Invalid token")

# ─── Flask App Setup ────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)
app.config["SECRET_KEY"] = SECRET_KEY

# ─── Auth Decorator ─────────────────────────────────────────────────────────────
from functools import wraps
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            raise Unauthorized("Missing Bearer token")
        token = header.split()[1]
        user = decode_jwt(token)
        g.user = user  # store user info in Flask global
        return f(*args, **kwargs)
    return decorated

# ─── Public Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/services")
def services():
    return render_template("data_services.html")

@app.route("/how-we-help")
def how_we_help():
    return render_template("how_we_help.html")

@app.route("/pricing")
def pricing():
    return render_template("pricing.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        # Validate form fields
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()
        recaptcha_token = request.form.get("g-recaptcha-response", "").strip()
        if not all([name, email, message]):
            flash("All fields are required.", "error")
            return redirect(request.url)
        if not recaptcha_token:
            flash("Please complete the CAPTCHA.", "error")
            return redirect(request.url)

        # Verify reCAPTCHA Enterprise
        try:
            access_token = get_oauth_token()
            resp = requests.post(
                f"https://recaptchaenterprise.googleapis.com/v1/projects/{PROJECT_ID}/assessments",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "event": {
                        "token": recaptcha_token,
                        "siteKey": RECAPTCHA_SITE_KEY,
                        "expectedAction": "submit_form"
                    }
                },
                timeout=5
            ).json()
            if not resp.get("tokenProperties", {}).get("valid"):
                flash("CAPTCHA validation failed.", "error")
                return redirect(request.url)
            if resp.get("riskAnalysis", {}).get("score", 0) < 0.5:
                flash("Suspicious activity detected.", "error")
                return redirect(request.url)
        except Exception as e:
            logger.error(f"CAPTCHA check error: {e}")
            flash("CAPTCHA verification error.", "error")
            return redirect(request.url)

        # Send email via SendGrid
        try:
            msg = Mail(
                from_email=SENDGRID_SENDER,
                to_emails=email,
                subject="New Contact from SynovaAI",
                html_content=f"<p><b>{name}</b> wrote:<br>{message}</p>"
            )
            msg.reply_to = ReplyTo(email)
            sg = SendGridAPIClient(SENDGRID_API_KEY)
            sg_resp = sg.send(msg)
            if sg_resp.status_code not in (200, 202):
                raise RuntimeError(f"SG error {sg_resp.status_code}")
        except Exception as e:
            logger.error(f"SendGrid error: {e}")
            flash("Error sending email.", "error")
            return redirect(request.url)

        flash("Message sent successfully!", "success")
        return redirect(url_for("contact"))

    return render_template("contact.html", recaptcha_site_key=RECAPTCHA_SITE_KEY)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email","").strip()
        pwd = request.form.get("password","")
        errors = []
        if len(pwd) < 8: errors.append("Min 8 chars")
        if not re.search(r"\d", pwd): errors.append("Must include digit")
        if not re.search(r"[A-Z]", pwd): errors.append("Must include uppercase")
        if not re.search(r"[!@#$%^&*]", pwd): errors.append("Must include special")
        if errors:
            for e in errors: flash(e,"error")
            return redirect(request.url)
        try:
            res = supabase.auth.sign_up({"email": email, "password": pwd})
            if res.get("error"):
                flash(res["error"]["message"], "error")
                return redirect(request.url)
            supabase.auth.update_user({"data":{"tier":"free"}}, user_id=res["user"]["id"])
            flash("Check your email to confirm.", "success")
            return redirect(url_for("login_page"))
        except Exception as e:
            logger.error(f"Register error: {e}")
            flash("Registration failed.", "error")
            return redirect(request.url)
    return render_template("register.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True)
    if not data or not data.get("email") or not data.get("password"):
        raise BadRequest("Email/password required")
    try:
        supa = requests.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            json={"email": data["email"], "password": data["password"]},
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json"
            },
            timeout=5
        )
        supa.raise_for_status()
        payload = supa.json()
        if "access_token" not in payload:
            return jsonify({"status":"error","message":payload.get("error_description","Login failed")}), 401
        user = payload.get("user",{})
        if not user.get("email_confirmed_at"):
            return jsonify({"status":"error","message":"Email not verified"}), 403
        token = generate_jwt(user["email"])
        return jsonify({
            "status":"success",
            "access_token": token,
            "tier": user.get("user_metadata",{}).get("tier","free")
        })
    except requests.RequestException as e:
        logger.error(f"Supabase auth error: {e}")
        abort(503, "Auth service unavailable")

@app.route("/forgot-password", methods=["GET","POST"])
def forgot_password():
    if request.method=="POST":
        email = request.form.get("email","").strip()
        if not email:
            flash("Email required","error")
            return redirect(request.url)
        try:
            res = supabase.auth.reset_password_for_email(email)
            if isinstance(res, dict) and res.get("error"):
                flash(res["error"]["message"],"error"); return redirect(request.url)
            flash("Check your email for reset link.","success")
            return redirect(url_for("login_page"))
        except Exception as e:
            logger.error(f"Forgot-password error: {e}")
            flash("Reset failed.","error")
            return redirect(request.url)
    return render_template("forgot_password.html")

@app.route("/update-password", methods=["GET","POST"])
def update_password():
    token = request.args.get("token","")
    email = request.args.get("email","")
    if not token or not email:
        flash("Missing token/email","error")
        return redirect(url_for("forgot_password"))
    if request.method=="POST":
        pwd = request.form.get("password","")
        if len(pwd)<8 or not re.search(r"\d",pwd) or not re.search(r"[A-Z]",pwd) or not re.search(r"[!@#$%^&*]",pwd):
            flash("Password rules not met","error")
            return render_template("update_password.html", token=token, email=email)
        try:
            otp = supabase.auth.verify_otp({"email":email,"token":token,"type":"recovery"})
            if "access_token" not in otp:
                flash("Invalid/expired token","error"); return redirect(url_for("forgot_password"))
            jwt_access = otp["access_token"]
            resp = requests.put(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={"Authorization":f"Bearer {jwt_access}", "apikey":SUPABASE_KEY,"Content-Type":"application/json"},
                json={"password":pwd},
                timeout=5
            )
            if resp.status_code!=200:
                flash("Password update failed","error"); return render_template("update_password.html",token=token,email=email)
            flash("Password updated!","success")
            return redirect(url_for("login_page"))
        except Exception as e:
            logger.error(f"Update-password error: {e}")
            flash("Error updating password","error")
            return redirect(url_for("forgot_password"))
    return render_template("update_password.html", token=token, email=email)

# ─── Protected Routes ──────────────────────────────────────────────────────────
@app.route("/forecasting-tool")
@token_required
def forecasting_tool():
    return render_template("forecasting_tool.html")

@app.route("/dashboard")
@token_required
def dashboard():
    user = g.user
    tier = user.get("tier","free")
    userid = user["email"].replace("@","_").replace(".","_")
    path = f"forecast_data/forecast_{userid}.json"
    data = []
    if os.path.exists(path):
        with open(path) as f: data = json.load(f)
    return render_template("dashboard.html", user=user, tier=tier, forecast_data=data)

# ─── Utility Static Routes & Error Handlers ──────────────────────────────────
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path,"static"), filename)

@app.after_request
def apply_security_headers(resp):
    resp.headers["Content-Security-Policy"] = "frame-ancestors 'self';"
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp

@app.errorhandler(401)
def handle_401(e):
    return redirect(url_for("login_page"))
@app.errorhandler(404)
def handle_404(e):
    return render_template("404.html"),404
@app.errorhandler(500)
def handle_500(e):
    return render_template("500.html"),500

# ─── Run Locally ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
