import os
import jwt as pyjwt  # ✅ Renaming to avoid conflicts
import datetime
import re
import requests
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, ReplyTo  
# ✅ Load environment variables from .env
load_dotenv()

# ✅ Initialize Supabase Client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ Load API Keys from Environment Variables
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_SENDER = os.getenv("SENDGRID_SENDER")
RECAPTCHA_SITE_KEY = os.getenv("RECAPTCHA_SITE_KEY")
RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY")

# ✅ Initialize Flask App
app = Flask(__name__, 
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "../templates")), 
            static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "../static")))

# ✅ Configure logging properly
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ✅ Use a Secure Flask Secret Key
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "super_secure_fallback_key")
app.config["SECRET_KEY"] = SECRET_KEY  # ✅ Set the secret key

# ✅ Function to Generate JWT Token
def generate_jwt(user_email):
    token = pyjwt.encode({
        "email": user_email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)  # Expires in 1 day
    }, SECRET_KEY, algorithm="HS256")
    return token

# ✅ Function to Verify JWT Token
def verify_jwt():
    """Extract and verify JWT from headers."""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or "Bearer" not in auth_header:
        return jsonify({"error": "Unauthorized"}), 401  # ✅ Return error JSON instead of None

    try:
        token = auth_header.split(" ")[1]  # Extract token
        decoded_token = pyjwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded_token  # ✅ Return user data
    except pyjwt.ExpiredSignatureError:
        return jsonify({"error": "JWT token expired"}), 401
    except pyjwt.InvalidTokenError:
        return jsonify({"error": "Invalid JWT token"}), 401
    
# ✅ Home Page
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Services Page
@app.route('/services')
def services():
    return render_template('data_services.html')

# ✅ How We Help Page
@app.route('/how-we-help')
def how_we_help():
    return render_template('how_we_help.html')

# ✅ Pricing Page
@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

# ✅ Success Page
@app.route('/success')
def success():
    return render_template('success.html')

@app.route("/contact", methods=["GET", "POST"])
def contact():
    recaptcha_site_key = os.getenv("RECAPTCHA_SITE_KEY")  # ✅ Pass to HTML

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message_body = request.form.get("message", "").strip()
        recaptcha_response = request.form.get("g-recaptcha-response", "").strip()  # ✅ Get reCAPTCHA token

        print(f"🔍 DEBUG: Received reCAPTCHA Token: {recaptcha_response}")  # Debug log

        # ✅ Validate Required Fields
        if not name or not email or not message_body:
            flash("⚠ Please fill in all fields.", "error")
            return redirect(request.referrer or url_for("contact"))

        # ✅ Verify reCAPTCHA
        recaptcha_secret = os.getenv("RECAPTCHA_SECRET_KEY")
        recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
        recaptcha_data = {"secret": recaptcha_secret, "response": recaptcha_response}
        recaptcha_result = requests.post(recaptcha_url, data=recaptcha_data).json()

        print(f"🔍 DEBUG: reCAPTCHA API Response: {recaptcha_result}")  # Log response

        if not recaptcha_result.get("success"):
            flash("❌ reCAPTCHA verification failed. Please try again.", "error")
            return redirect(request.referrer or url_for("contact"))

        # ✅ (Optional) Check reCAPTCHA v3 Score
        if "score" in recaptcha_result and recaptcha_result["score"] < 0.5:
            flash("⚠ reCAPTCHA score too low, suspected bot activity.", "error")
            return redirect(request.referrer or url_for("contact"))

        # ✅ Construct Email with Reply-To Header
        message = Mail(
            from_email=os.getenv("SENDGRID_SENDER"),  # Must be a verified sender email
            to_emails="jslad13@gmail.com",  # Replace with actual receiving email
            subject="New Contact Form Submission - SynovaAI",
            html_content=f"""
                <p><strong>Name:</strong> {name}</p>
                <p><strong>Email:</strong> {email}</p>
                <p><strong>Message:</strong> {message_body}</p>
            """
        )

        message.reply_to = ReplyTo(email)  # ✅ Allow direct reply to sender

        try:
            sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
            response = sg.send(message)

            print(f"📨 DEBUG: SendGrid Response Code: {response.status_code}")
            print(f"📨 DEBUG: SendGrid Response Body: {response.body.decode('utf-8') if response.body else 'No Content'}")

            # ✅ Handle API Responses
            if response.status_code in [200, 202]:
                flash("✅ Your message has been sent successfully!", "success")
            else:
                flash(f"❌ Failed to send message. Error {response.status_code}.", "error")

        except Exception as e:
            print(f"❌ SendGrid Error: {e}")
            flash("❌ Error sending email. Please try again later.", "error")

        return redirect(request.referrer or url_for("contact"))

    return render_template("contact.html", recaptcha_site_key=recaptcha_site_key)

# ✅ Data Services Page
@app.route('/data-services')
def data_services():
    return render_template('data_services.html')

@app.route('/forecasting-tool')
def forecasting_tool():
    user_info = verify_jwt()
    if not user_info:
        flash("❌ You must be logged in to access the forecasting tool.", "error")
        return redirect(url_for('login'))
    
    return render_template('forecasting_tool.html')

# ✅ Self-Service Insights Page
@app.route('/self-service-insights')
def self_service_insights():
    return render_template('self_service_insights.html')

# ✅ Dashboard Route (Protected with JWT)
def token_required(f):
    """Decorator to ensure JWT authentication is required for routes."""
    def decorated_function(*args, **kwargs):
        token = request.cookies.get("access_token") or request.headers.get("Authorization")

        if not token:
            logger.warning("❌ Unauthorized: No JWT token provided.")
            return redirect(url_for('login_page'))

        try:
            if token.startswith("Bearer "):
                token = token.split(" ")[1]  # Extract token if "Bearer <token>"

            # ✅ Decode and validate the JWT
            payload = pyjwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user = payload  # Store user info in request context

        except pyjwt.ExpiredSignatureError:
            logger.warning("❌ Unauthorized: JWT token expired.")
            return redirect(url_for('login_page'))
        except pyjwt.InvalidTokenError:
            logger.warning("❌ Unauthorized: Invalid JWT token.")
            return redirect(url_for('login_page'))

        return f(*args, **kwargs)
    
    return decorated_function

@app.route('/dashboard', methods=['GET'])
@token_required
def dashboard():
    try:
        user_info = request.user  # Get user info from JWT
        user_plan = user_info.get("tier", "free")  # Default to "free" if missing

        logger.debug(f"✅ User {user_info['email']} accessed dashboard. Subscription: {user_plan}")

        return render_template('dashboard.html', user_info=user_info, user_plan=user_plan)

    except Exception as e:
        logger.exception("🔥 Error loading dashboard")
        return render_template('500.html'), 500
    
# ✅ Register Route (Uses Supabase Auth)
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        errors = []

        # ✅ Password validation rules
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long.")
        if not re.search(r"\d", password):
            errors.append("Password must contain at least one number.")
        if not re.search(r"[A-Z]", password):
            errors.append("Password must contain at least one uppercase letter.")
        if not re.search(r"[!@#$%^&*]", password):
            errors.append("Password must contain at least one special character (!@#$%^&*).")

        # ✅ Show all validation errors at once
        if errors:
            for error in errors:
                flash(error, "error")
            return redirect(url_for("register"))

        # ✅ Attempt to register the user
        try:
            response = supabase.auth.sign_up({
                "email": email,
                "password": password
            })

            # ✅ Check if signup failed
            if "error" in response and response["error"]:
                flash(f"Registration failed: {response['error']['message']}", "error")
                return redirect(url_for("register"))

            # ✅ Store default 'tier' metadata
            user_id = response["user"]["id"]
            supabase.auth.update_user({
                "data": {"tier": "free"}
            }, user_id=user_id)

            flash("Check your email to confirm your account.", "success")
            return redirect(url_for("login"))

        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for("register"))

    return render_template('register.html')

@app.route('/update-password', methods=['GET', 'POST'])
def update_password():
    """Handles password reset with Supabase using JWT authentication"""

    access_token = request.args.get('token')  # Capture JWT token from URL
    user_email = request.args.get("email")  # Get email from request

    print(f"🔍 Received Token: {access_token}")  # Debugging
    print(f"🔍 Received Email: {user_email}")  # Debugging

    # ✅ Ensure both email and token are provided
    if not user_email or not access_token:
        flash("❌ Email and token are required for password reset.", "error")
        return redirect(url_for("forgot_password"))

    if request.method == 'POST':
        password = request.form.get('password')

        # ✅ Validate password complexity
        if not password or len(password) < 8:
            flash("❌ Password must be at least 8 characters long.", "error")
            return render_template("update_password.html", token=access_token, email=user_email)
        if not re.search(r"\d", password):
            flash("❌ Password must contain at least one digit.", "error")
            return render_template("update_password.html", token=access_token, email=user_email)
        if not re.search(r"[A-Z]", password):
            flash("❌ Password must contain at least one uppercase letter.", "error")
            return render_template("update_password.html", token=access_token, email=user_email)
        if not re.search(r"[!@#$%^&*]", password):  # Ensure at least one special character
            flash("❌ Password must contain at least one special character (!@#$%^&*).", "error")
            return render_template("update_password.html", token=access_token, email=user_email)

        try:
            print(f"🔍 Verifying OTP with token: {access_token}")

            # ✅ Step 1: Verify OTP (JWT Recovery Token)
            otp_response = supabase.auth.verify_otp({
                "email": user_email,
                "token": access_token,
                "type": "recovery"  # ✅ Use "recovery" to verify reset token
            })

            print("🔍 OTP Verification Response:", otp_response)  # Debugging

            # ✅ Ensure response contains a valid JWT access token
            if "access_token" not in otp_response:
                flash("❌ Verification failed. The reset token may be invalid or expired.", "error")
                return redirect(url_for("forgot_password"))

            jwt_access_token = otp_response["access_token"]  # Extract JWT

            print(f"✅ Authenticated JWT: {jwt_access_token}")

            # ✅ Step 2: Update Password Using JWT
            headers = {
                "Authorization": f"Bearer {jwt_access_token}",  # Use JWT for authentication
                "apikey": SUPABASE_KEY,
                "Content-Type": "application/json",
            }

            update_response = requests.put(
                f"{SUPABASE_URL}/auth/v1/user",
                json={"password": password},
                headers=headers,
            )

            update_data = update_response.json()
            print("🔍 Password Update Response:", update_data)  # Debugging

            # ✅ Ensure password update was successful
            if update_response.status_code != 200:
                error_message = update_data.get("message", "Unknown error")
                flash(f"❌ Password update failed: {error_message}", "error")
                return render_template("update_password.html", token=access_token, email=user_email)

            flash("✅ Password updated successfully! You can now log in.", "success")
            return redirect(url_for("login"))

        except Exception as e:
            print("🔥 Exception Occurred:", str(e))  # Debugging
            flash(f"❌ Error updating password: {str(e)}", "error")
            return redirect(url_for("forgot_password"))  # Redirect on failure

    return render_template("update_password.html", token=access_token, email=user_email)

# ✅ Login Route (JWT-Based)
@app.route('/login', methods=['GET'])  # ✅ Show login page
def login_page():
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def login():
    try:
        logger.debug("🔍 Received login request.")

        if not request.is_json:
            return jsonify({"status": "error", "message": "Unsupported Media Type: Use 'application/json'"}), 415

        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"status": "error", "message": "Missing email or password"}), 400

        logger.debug(f"🔍 Attempting login for {email}")

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }

        supabase_response = requests.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            json={"email": email, "password": password},
            headers=headers
        )

        logger.debug(f"🔍 Supabase Raw Response: {supabase_response.text}")

        try:
            supabase_data = supabase_response.json()
        except ValueError:
            logger.error("❌ Supabase did not return valid JSON.")
            return jsonify({"status": "error", "message": "Invalid response from authentication server"}), 500

        # ✅ Handle authentication errors
        if supabase_response.status_code != 200 or "access_token" not in supabase_data:
            error_message = supabase_data.get("error_description", supabase_data.get("error", "Invalid login credentials"))
            logger.warning(f"❌ Authentication failed: {error_message}")
            return jsonify({"status": "error", "message": error_message}), 401

        user_data = supabase_data.get("user", {})
        user_metadata = user_data.get("user_metadata", {})

        # ✅ Check if email is verified before issuing JWT
        if not user_data.get("email_confirmed_at"):
            logger.warning("❌ Login failed: Email not verified.")
            return jsonify({"status": "error", "message": "Email not verified. Please check your inbox."}), 403

        user_tier = user_metadata.get("tier", "free")

        # ✅ Generate JWT Token
        jwt_token = generate_jwt(user_data.get("email"))

        logger.debug(f"✅ Login successful. JWT issued.")

        return jsonify({
            "status": "success",
            "redirect": "/dashboard",
            "access_token": jwt_token,
            "tier": user_tier
        })

    except requests.exceptions.RequestException as req_error:
        logger.error(f"🔥 Network error during Supabase request: {str(req_error)}")
        return jsonify({"status": "error", "message": "Authentication service unavailable"}), 503

    except Exception as e:
        logger.exception("🔥 Unexpected server error during login process.")
        return jsonify({"status": "error", "message": "Internal server error"}), 500

@app.route('/api/verify-token', methods=['GET'])
def verify_token():
    """Verifies if the provided JWT token is valid"""
    auth_header = request.headers.get("Authorization")

    if not auth_header or "Bearer" not in auth_header:
        logger.warning("❌ No JWT token provided.")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    try:
        token = auth_header.split(" ")[1]  # Extract token after "Bearer"
        decoded_token = pyjwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        
        logger.debug("✅ JWT verified successfully.")
        return jsonify({"status": "success", "user": decoded_token})

    except pyjwt.ExpiredSignatureError:
        logger.warning("❌ JWT token expired.")
        return jsonify({"status": "error", "message": "Token expired"}), 401

    except pyjwt.InvalidTokenError:
        logger.warning("❌ Invalid JWT token.")
        return jsonify({"status": "error", "message": "Invalid token"}), 401

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')

        if not email:
            flash("❌ Email is required to reset your password.", "error")
            return redirect(url_for("forgot_password"))

        try:
            print(f"🔍 Sending password reset request for: {email}")  # Debugging Log

            # ✅ Supabase Password Reset Request
            response = supabase.auth.reset_password_for_email(email)

            # ✅ Validate Supabase Response
            if isinstance(response, dict) and "error" in response:
                error_message = response["error"].get("message", "Unknown error")
                print(f"❌ Supabase Error: {error_message}")  # Debugging Log
                flash(f"❌ Password reset failed: {error_message}", "error")
                return redirect(url_for("forgot_password"))

            flash("📩 Check your email for a password reset link.", "success")
            return redirect(url_for('login'))

        except Exception as e:
            print(f"🔥 Exception in forgot-password: {str(e)}")  # Debugging Log
            flash("❌ An error occurred. Please try again later.", "error")
            return redirect(url_for("forgot_password"))

    return render_template('forgot_password.html')

# ✅ Logout Route (Clears Frontend JWT)
@app.route('/logout')
def logout():
    return redirect(url_for("home"))

# ✅ Check Auth Route (JWT-Based)
@app.route('/check-auth')
def check_auth():
    token = request.args.get("token")  # Get token from request

    if not token:
        print("❌ No token provided")  # Debugging
        return jsonify({"error": "Unauthorized"}), 401

    user_email = verify_jwt(token)
    
    if not user_email:
        print("❌ Invalid or expired token")  # Debugging
        return jsonify({"error": "Invalid token"}), 401

    print(f"✅ Authenticated User: {user_email}")  # Debugging
    return jsonify({"status": "authenticated", "user_email": user_email})

# ✅ Stripe Payment Route
@app.route('/checkout/pro')
def checkout_pro():
    return redirect("https://buy.stripe.com/eVa00j89SeIM9hKeUU")

# ✅ Serve Static Files (JS, Images, General Static Files)
@app.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(os.path.join(app.root_path, "static"), filename)

# ✅ Serve CSS Files Separately (Optional)
@app.route('/css/<path:filename>')
def serve_css_files(filename):
    return send_from_directory(os.path.join(app.root_path, "static", "css"), filename)

# ✅ Security Headers
@app.after_request
def apply_security_headers(response):
    response.headers["Content-Security-Policy"] = "frame-ancestors 'self';"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# ✅ Error Handling
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# ✅ Run App Locally (For Debugging)
if __name__ == "__main__":
    app.run(debug=True)
