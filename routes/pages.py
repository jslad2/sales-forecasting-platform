from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)

@pages_bp.route("/")
def home():
    return render_template("index.html")

@pages_bp.route("/data-services")
def data_services():
    return render_template("data_services.html")

@pages_bp.route("/how-we-help")
def how_we_help():
    return render_template("how_we_help.html")

@pages_bp.route("/pricing")
def pricing():
    return render_template("pricing.html")

@pages_bp.route("/success")
def success():
    return render_template("success.html")
