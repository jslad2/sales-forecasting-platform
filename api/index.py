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

# ✅ Forecasting Tool Page (Restricted)
@app.route('/forecasting-tool')
def forecasting_tool():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    user = conn.execute("SELECT tier FROM users WHERE email = ?", (session['user'],)).fetchone()
    conn.close()

    if user and user['tier'] == 'pro':
        return render_template('forecasting_tool.html', pro_user=True)
    else:
        return render_template('forecasting_tool.html', pro_user=False)

# ✅ Sales Dashboard Page (Restricted)
@app.route('/sales-dashboard')
def sales_dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    user = conn.execute("SELECT tier FROM users WHERE email = ?", (session['user'],)).fetchone()
    conn.close()

    if user and user['tier'] == 'pro':
        return render_template('sales_dashboard.html', pro_user=True)
    else:
        return render_template('sales_dashboard.html', pro_user=False)

# ✅ Self-Service Insights Page
@app.route('/self-service-insights')
def self_service_insights():
    return render_template('self_service_insights.html')

# ✅ Dashboard Route (Checks Free vs. Pro)
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT tier FROM users WHERE email = ?", (session['user'],))
    user = cursor.fetchone()
    conn.close()

    if user and user['tier'] == 'pro':
        return render_template('dashboard.html', pro_user=True)
    else:
        return render_template('dashboard.html', pro_user=False)
    
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
            if response.user is None:
                flash("Registration failed.")
                return redirect(url_for("register"))

            flash("Check your email to confirm your account.")
            return redirect(url_for("login"))

        except Exception as e:
            flash(f"Error: {str(e)}")
            return redirect(url_for("register"))

    return render_template('register.html')

@app.route('/update-password', methods=['GET', 'POST'])
def update_password():
    """Handles password reset with Supabase"""
    
    access_token = request.args.get('token')  # Capture the token from the URL

    if not access_token:
        flash("Invalid or missing token. Please request a new password reset.", "error")
        return redirect(url_for("login"))

    if request.method == 'POST':
        password = request.form.get('password')

        # ✅ Validate password complexity
        if not password or len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
            return redirect(url_for("update_password", token=access_token))
        if not re.search(r"\d", password):
            flash("Password must contain at least one digit.", "error")
            return redirect(url_for("update_password", token=access_token))
        if not re.search(r"[A-Z]", password):
            flash("Password must contain at least one uppercase letter.", "error")
            return redirect(url_for("update_password", token=access_token))

        try:
            # ✅ Authenticate the session using the reset token
            session_response = supabase.auth.exchange_code_for_session(access_token)

            if not session_response or "user" not in session_response:
                flash("Invalid or expired reset token.", "error")
                return redirect(url_for("update_password", token=access_token))

            user_email = session_response["user"]["email"]  # ✅ Extract the email

            # ✅ Update the password (Pass both email & password)
            user_update_response = supabase.auth.update_user({
                "email": user_email,  # 🔹 REQUIRED FIELD
                "password": password
            })

            if "error" in user_update_response and user_update_response["error"]:
                flash(f"Error: {user_update_response['error']['message']}", "error")
                return redirect(url_for("update_password", token=access_token))

            flash("Password updated successfully! You can now log in.", "success")
            return redirect(url_for("login"))

        except Exception as e:
            flash(f"Error updating password: {str(e)}", "error")
            return redirect(url_for("update_password", token=access_token))

    return render_template("update_password.html", token=access_token)


# ✅ Login Route (Uses Supabase Auth)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})

            if not response.user:
                flash("Invalid email or password. Please try again.", "error")
                return redirect(url_for("login"))

            session['user'] = email
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))

        except Exception as e:
            flash(f"Login failed: {str(e)}", "error")
            return redirect(url_for("login"))

    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')

        try:
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
    session.pop('user', None)
    return redirect(url_for('home'))

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
