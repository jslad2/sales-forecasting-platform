import os
import re
from flask import Blueprint, request, jsonify, flash, redirect, url_for, render_template
from werkzeug.exceptions import BadRequest, Unauthorized
import requests
from utils import supabase, generate_jwt, decode_jwt, get_oauth_token, PROJECT_ID

# Load Supabase config from env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)
    if not data or not data.get("email") or not data.get("password"):
        raise BadRequest("Email & password required")

    # Authenticate with Supabase
    supa_resp = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        json={
            "email": data["email"],
            "password": data["password"]
        },
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
    )
    payload = supa_resp.json()
    if supa_resp.status_code != 200 or "access_token" not in payload:
        return jsonify({
            "status": "error",
            "message": payload.get("error_description", "Login failed")
        }), 401

    # Ensure email is confirmed
    if not payload["user"].get("email_confirmed_at"):
        return jsonify({
            "status": "error",
            "message": "Email not verified"
        }), 403

    # Issue our own JWT
    token = generate_jwt(payload["user"]["email"])
    tier  = payload["user"]["user_metadata"].get("tier", "free")

    return jsonify({
        "status":       "success",
        "access_token": token,
        "tier":         tier
    })


@auth_bp.route("/verify-token", methods=["GET"])
def verify_token():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise Unauthorized("Missing token")
    token = auth_header.split()[1]
    user = decode_jwt(token)
    return jsonify({
        "status": "success",
        "user":   user
    })


@auth_bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    email = request.form.get("email", "").strip()
    if not email:
        flash("Email required", "error")
        return redirect(url_for("pages.home"))

    result = supabase.auth.reset_password_for_email(email)
    if isinstance(result, dict) and result.get("error"):
        flash(result["error"]["message"], "error")
        return redirect(url_for("pages.home"))

    flash("Check your email for a password reset link", "success")
    return redirect(url_for("pages.home"))


@auth_bp.route("/update-password", methods=["GET", "POST"])
def update_password():
    token = request.args.get("token", "")
    email = request.args.get("email", "")

    if request.method == "POST":
        pwd = request.form.get("password", "")
        # enforce complexity
        if (len(pwd) < 8 or
            not re.search(r"\d", pwd) or
            not re.search(r"[A-Z]", pwd) or
            not re.search(r"[!@#$%^&*]", pwd)):
            flash("Password must be ≥8 chars, include a digit, uppercase and special char", "error")
            return render_template("update_password.html", token=token, email=email)

        # verify OTP
        otp_resp = supabase.auth.verify_otp({
            "email": email,
            "token": token,
            "type":  "recovery"
        })
        if "access_token" not in otp_resp:
            flash("Invalid or expired token", "error")
            return redirect(url_for("pages.home"))

        jwt_access = otp_resp["access_token"]
        update_resp = requests.put(
            f"{SUPABASE_URL}/auth/v1/user",
            json={"password": pwd},
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {jwt_access}",
                "Content-Type":  "application/json"
            }
        )
        if update_resp.status_code != 200:
            flash("Password update failed", "error")
            return render_template("update_password.html", token=token, email=email)

        flash("Password updated successfully!", "success")
        return redirect(url_for("pages.home"))

    # GET: show form
    return render_template("update_password.html", token=token, email=email)
