from flask import Blueprint, render_template, request, flash, redirect, url_for

pages_bp = Blueprint("pages", __name__)

@pages_bp.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@pages_bp.route("/data-services", methods=["GET"])
def data_services():
    return render_template("data_services.html")

@pages_bp.route("/how-we-help", methods=["GET"])
def how_we_help():
    return render_template("how_we_help.html")

@pages_bp.route("/pricing", methods=["GET"])
def pricing():
    return render_template("pricing.html")

@pages_bp.route("/success", methods=["GET"])
def success():
    return render_template("success.html")

@pages_bp.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        # Basic form validation
        name    = request.form.get("name", "").strip()
        email   = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        if not (name and email and message):
            flash("All fields are required.", "error")
            return redirect(url_for("pages.contact"))

        # TODO: insert your reCAPTCHA & email‐sending logic here
        # e.g. verify_recaptcha(token), send_email(name, email, message), etc.

        flash("Your message has been sent successfully!", "success")
        return redirect(url_for("pages.success"))

    # GET
    return render_template("contact.html")
