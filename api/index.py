import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import dotenv
import subprocess

def installed_packages():
    result = subprocess.run(["pip", "list"], capture_output=True, text=True)
    return result.stdout

print("=== INSTALLED PACKAGES ===")
print(installed_packages())


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

@app.route("/debug-packages")
def debug_packages():
    result = subprocess.run(["pip", "list"], capture_output=True, text=True)
    return jsonify({"installed_packages": result.stdout})

# ✅ Run App Locally (For Debugging)
if __name__ == "__main__":
    app.run(debug=True)
