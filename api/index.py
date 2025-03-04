import os
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client, Client
from dotenv import load_dotenv
import re

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Initialize Supabase Client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ✅ Initialize Flask App (Ensure Correct Paths)
app = Flask(__name__, 
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "../templates")), 
            static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "../static")))

app.secret_key = "your_secret_key"  # Change this for security

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

# ✅ Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# ✅ Data Services Page
@app.route('/data-services')
def data_services():
    return render_template('data_services.html')

@app.route('/forecasting-tool')
def forecasting_tool():
    if 'user_email' not in session:
        flash("❌ You must be logged in to access the forecasting tool.", "error")
        return redirect(url_for('login'))

    # Retrieve user subscription level from Supabase
    user_email = session.get("user_email")
    user_plan = session.get("user_plan", "free")

    return render_template('forecasting_tool.html', user_plan=user_plan)

# ✅ Self-Service Insights Page
@app.route('/self-service-insights')
def self_service_insights():
    return render_template('self_service_insights.html')

# ✅ Dashboard Route
@app.route('/dashboard')
def dashboard():
    if "user_email" not in session:
        return redirect(url_for("login"))  # 🔐 Redirect if NOT logged in

    return render_template("dashboard.html", 
                           user_email=session["user_email"], 
                           user_plan=session["user_plan"])

    
# ✅ Register Route (Uses Supabase Auth)
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # ✅ Password validation rules
        if len(password) < 8:  # Minimum length
            flash("Password must be at least 8 characters long.")
            return redirect(url_for("register"))

        if not re.search(r"\d", password):  # At least one digit
            flash("Password must contain at least one number.")
            return redirect(url_for("register"))

        if not re.search(r"[A-Z]", password):  # At least one uppercase letter
            flash("Password must contain at least one uppercase letter.")
            return redirect(url_for("register"))

        if not re.search(r"[!@#$%^&*]", password):  # At least one special character
            flash("Password must contain at least one special character (!@#$%^&*).")
            return redirect(url_for("register"))

        # ✅ Attempt to register the user
        try:
            response = supabase.auth.sign_up({"email": email, "password": password})

            if "error" in response and response["error"]:
                flash(f"Registration failed: {response['error']['message']}", "error")
                return redirect(url_for("register"))

            flash("Check your email to confirm your account.", "success")
            return redirect(url_for("login"))

        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for("register"))

    return render_template('register.html')

@app.route('/update-password', methods=['GET', 'POST'])
def update_password():
    """Handles password reset with Supabase"""

    access_token = request.args.get('token')  # Capture the token from the URL
    user_email = request.args.get("email") or session.get("reset_email")  # Retrieve email from session if missing

    print(f"🔍 Received Token: {access_token}")  # Debugging
    print("🔍 Request Arguments:", request.args)
    print("🔍 Session Data:", session)

    # ⚠️ Handle missing email properly instead of redirecting infinitely
    if not user_email:
        flash("Email is required for password reset.", "error")
        return redirect(url_for("forgot_password"))

    if request.method == 'POST':
        password = request.form.get('password')

        # ✅ Validate password complexity
        if not password or len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
            return render_template("update_password.html", token=access_token)
        if not re.search(r"\d", password):
            flash("Password must contain at least one digit.", "error")
            return render_template("update_password.html", token=access_token)
        if not re.search(r"[A-Z]", password):
            flash("Password must contain at least one uppercase letter.", "error")
            return render_template("update_password.html", token=access_token)

        try:
            print(f"🔍 Attempting sign-in with recovery token: {access_token}")

            # ✅ Authenticate user with Supabase using recovery token
            session_response = supabase.auth.sign_in_with_otp({
                "email": user_email,
                "token": access_token,
                "type": "recovery"  # ✅ Explicitly define type to ensure correct authentication
            })

            # 🔍 Debugging: Print the full session response
            print("🔍 Full Session Response:", session_response)

            # ✅ Ensure response is valid and extract user email
            if not isinstance(session_response, dict):
                flash("Invalid response format from authentication. Please try again.", "error")
                return redirect(url_for("forgot_password"))

            if "user" not in session_response or not session_response["user"]:
                flash("Authentication failed. Your reset token may be invalid or expired.", "error")
                return redirect(url_for("forgot_password"))

            authenticated_user = session_response["user"]

            if "email" not in authenticated_user:
                flash("Session authentication failed. Please request a new password reset.", "error")
                return redirect(url_for("forgot_password"))

            user_email = authenticated_user["email"]  # ✅ Keep the validated email

            print(f"✅ Authenticated User Email: {user_email}")

            # ✅ Update password after successful authentication
            user_update_response = supabase.auth.update_user({"password": password})

            # 🔍 Debugging: Print the response from the password update
            print("🔍 User Update Response:", user_update_response)

            # ✅ Ensure the update was successful
            if not isinstance(user_update_response, dict) or "error" in user_update_response:
                error_message = user_update_response.get("error", {}).get("message", "Unknown error")
                flash(f"Password update failed: {error_message}", "error")
                return render_template("update_password.html", token=access_token)

            flash("Password updated successfully! You can now log in.", "success")
            return redirect(url_for("login"))

        except Exception as e:
            print("🔥 Exception Occurred:", str(e))  # Debugging
            flash(f"Error updating password: {str(e)}", "error")
            return redirect(url_for("forgot_password"))  # Redirect to forgot password on failure

    return render_template("update_password.html", token=access_token)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        print(f"🔍 Attempting login for {email}")  # Debugging

        try:
            user_response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            print(f"🔍 Supabase Response: {user_response}")  # Debugging

            if user_response and "user" in user_response:
                user_data = user_response["user"]

                # ✅ Store session data correctly
                session["user_email"] = user_data["email"]
                session["user_id"] = user_data["id"]
                session["session_id"] = os.urandom(24).hex()  # Unique session ID
                session["access_token"] = user_response["access_token"]  # Store the token
                session["refresh_token"] = user_response["refresh_token"]

                print(f"✅ Login successful! Session ID: {session['session_id']}")  # Debugging
                return jsonify({"status": "success", "redirect": "/dashboard"})  # Respond with success

            else:
                print("❌ Supabase returned invalid credentials")  # Debugging
                return jsonify({"status": "error", "message": "Invalid login credentials"}), 401

        except Exception as e:
            print(f"🔥 Login error: {str(e)}")  # Debugging
            return jsonify({"status": "error", "message": str(e)}), 500

    return render_template("login.html")

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')

        try:
            # ✅ Store email in session for later retrieval
            session["reset_email"] = email  

            # ✅ Supabase Password Reset Request
            response = supabase.auth.reset_password_for_email(email)

            flash("Check your email for a password reset link.", "success")
            return redirect(url_for('login'))

        except Exception as e:
            flash(f"Error: {str(e)}", "error")
            return redirect(url_for("forgot_password"))

    return render_template('forgot_password.html')

# ✅ Logout Route
@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

@app.route('/check-auth')
def check_auth():
    print(f"🔍 Stored session: {session.get('session_id')}")  # Debugging

    session_id = request.args.get("session_id")

    if not session_id or session_id != session.get("session_id"):
        print("❌ Invalid session detected")  # Debugging
        return jsonify({"error": "Invalid session"}), 401  # Unauthorized

    return jsonify({"status": "authenticated", "user_email": session["user_email"]})

# ✅ Stripe Payment Route
@app.route('/checkout/pro')
def checkout_pro():
    return redirect("https://buy.stripe.com/eVa00j89SeIM9hKeUU")

# ✅ Serve Static Files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, "static"), filename)

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(app.root_path, "static/css"), filename)

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
