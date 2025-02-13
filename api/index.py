from flask import Flask, render_template, send_from_directory

# Initialize Flask app
app = Flask(__name__, static_folder="static")

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/services')
def services():
    return render_template('data_services.html')

@app.route('/how-we-help')
def how_we_help():
    return render_template('how_we_help.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/data-services')
def data_services():
    return render_template('data_services.html')

@app.route('/forecasting-tool')
def forecasting_tool():
    return render_template('forecasting_tool.html')

@app.route('/sales-dashboard')
def sales_dashboard():
    return render_template('sales_dashboard.html')

@app.route('/self-service-insights')
def self_service_insights():
    return render_template('self_service_insights.html')

# Serve static files (CSS, JS, images)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Serve CSS files from the CSS folder
@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('css', filename)

# Security headers for embedding Streamlit in iframes
@app.after_request
def apply_security_headers(response):
    response.headers["Content-Security-Policy"] = "frame-ancestors 'self';"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# Error handling
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# ✅ IMPORTANT: Expose `app` for Vercel (This is what Vercel expects)
if __name__ == "__main__":
    app.run()
