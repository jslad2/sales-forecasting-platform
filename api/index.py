import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client, Client
from dotenv import load_dotenv
import subprocess

installed_packages = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
print(installed_packages.stdout)


load_dotenv()  # Load environment variables from .env file

# ✅ Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ✅ Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ Initialize Flask App (Ensure Correct Paths)
app = Flask(__name__, 
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "../templates")), 
            static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "../static")))

app.secret_key = "your_secret_key"  # Change this for security

# ✅ Database Connection
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

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

# ✅ Register Route (Signup with Supabase)
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # ✅ Create user in Supabase Auth
        try:
            response = supabase.auth.sign_up({"email": email, "password": password})
            if response.user:
                # ✅ Insert user into 'users' table with 'free' tier
                supabase.table("users").insert({"email": email, "password_hash": hashed_password, "tier": "free"}).execute()

                session['user'] = email  # Log in user after sign-up
                return redirect(url_for('dashboard'))
            else:
                return "Error: Unable to create user."
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template('register.html')

# ✅ Login Route (Sign in with Supabase)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # ✅ Authenticate user with Supabase
        try:
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if response.user:
                session['user'] = email
                return redirect(url_for('dashboard'))
            else:
                return "Invalid login. <a href='/login'>Try again</a>."
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template('login.html')

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
