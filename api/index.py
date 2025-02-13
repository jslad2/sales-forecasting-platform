from flask import Flask, render_template, send_from_directory
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response

# Initialize Flask app
app = Flask(__name__, static_folder="static")

# Routes
@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/services')
def services():
    """Render the services page."""
    return render_template('data_services.html')

@app.route('/how-we-help')
def how_we_help():
    """Render the how-we-help page."""
    return render_template('how_we_help.html')

@app.route('/contact')
def contact():
    """Render the contact page."""
    return render_template('contact.html')

@app.route('/data-services')
def data_services():
    """Render the data services page."""
    return render_template('data_services.html')

@app.route('/forecasting-tool')
def forecasting_tool():
    """Render the forecasting tool page with embedded Streamlit app."""
    return render_template('forecasting_tool.html')

@app.route('/sales-dashboard')
def sales_dashboard():
    """Render the sales dashboard page with embedded Streamlit app."""
    return render_template('sales_dashboard.html')

@app.route('/self-service-insights')
def self_service_insights():
    """Render the self-service insights page."""
    return render_template('self_service_insights.html')

# Serve static files (CSS, JS, images)
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files from the static folder."""
    return send_from_directory('static', filename)

# Serve CSS files from the CSS folder
@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files from the CSS folder."""
    return send_from_directory('css', filename)

# Security headers for embedding Streamlit in iframes
@app.after_request
def apply_security_headers(response):
    """
    Apply security headers to allow embedding Streamlit apps in iframes.
    """
    response.headers["Content-Security-Policy"] = "frame-ancestors *;"
    response.headers["X-Frame-Options"] = "ALLOWALL"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

# Convert Flask app to a WSGI-compatible function for Vercel
def handler(event, context):
    """Custom WSGI handler for Vercel."""
    request = Request(event)
    response = Response.from_app(app, request.environ)
    return response(environ=request.environ, start_response=context)
