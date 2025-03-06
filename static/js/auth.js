document.addEventListener("DOMContentLoaded", function () {
    console.log("✅ auth.js is loaded and running!");

    const loginForm = document.querySelector("#login-form");

    if (!loginForm) {
        console.warn("❌ loginForm not found. JavaScript might not be running!");
        return;
    }

    console.log("✅ Found login form! Adding event listener...");

    loginForm.addEventListener("submit", async function (e) {
        e.preventDefault(); // ✅ Prevents default form submission.
        console.log("✅ Form submission prevented, sending POST request...");

        const email = document.querySelector("#email").value.trim();
        const password = document.querySelector("#password").value.trim();
        const loginButton = document.querySelector("#login-button");

        if (!email || !password) {
            alert("⚠️ Please enter both email and password.");
            return;
        }

        if (loginButton) {
            loginButton.disabled = true;
            loginButton.textContent = "Logging in...";
        }

        try {
            console.log("🔍 Sending login request...");

            const response = await fetch("/api/login", {  
                method: "POST",  
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password })
            });

            const data = await response.json();
            console.log("✅ Login Response:", data);

            if (response.ok && data.status === "success") {
                alert("✅ Login successful! Redirecting...");

                // ✅ Store JWT token in localStorage
                localStorage.setItem("access_token", data.access_token);

                // ✅ Verify token before redirecting
                const tokenValid = await verifyToken();
                if (tokenValid) {
                    window.location.href = data.redirect; // ✅ Redirect to dashboard
                } else {
                    alert("⚠️ Session expired. Please log in again.");
                    window.location.href = "/login";
                }
            } else {
                alert(`❌ Login failed: ${data.message || "Invalid credentials"}`);
            }
        } catch (error) {
            console.error("🔥 Login Error:", error);
            alert("❌ An error occurred. Please try again.");
        } finally {
            if (loginButton) {
                loginButton.disabled = false;
                loginButton.textContent = "Login";
            }
        }
    });

    // ✅ Auto-check authentication on protected pages
    if (window.location.pathname === "/dashboard") {
        checkAuth();
    }
});

/**
 * ✅ Function to check if the user is authenticated (JWT exists)
 */
async function checkAuth() {
    const token = localStorage.getItem("access_token");

    if (!token) {
        console.warn("❌ No JWT token found. Redirecting to login.");
        alert("⚠️ Session expired. Please log in again.");
        window.location.href = "/login";
        return;
    }

    const tokenValid = await verifyToken();
    if (!tokenValid) {
        console.warn("❌ Token is invalid. Logging out.");
        alert("⚠️ Session expired. Please log in again.");
        localStorage.removeItem("access_token");
        window.location.href = "/login";
    } else {
        console.log("✅ JWT verified. Access granted.");
    }
}

/**
 * ✅ Function to verify the stored JWT token with the backend
 */
async function verifyToken() {
    const token = localStorage.getItem("access_token");

    if (!token) {
        console.warn("❌ No JWT token found.");
        return false;
    }

    try {
        console.log("🔍 Verifying JWT token...");
        const response = await fetch("/api/verify-token", {
            method: "GET",
            headers: { "Authorization": `Bearer ${token}` }
        });

        if (!response.ok) {
            console.warn("❌ Invalid or expired token.");
            return false;
        }

        console.log("✅ Token is valid.");
        return true;
    } catch (error) {
        console.error("🔥 Error verifying token:", error);
        return false;
    }
}

/**
 * ✅ Wrapper function to send authenticated API requests
 */
async function fetchWithAuth(url, options = {}) {
    const token = localStorage.getItem("access_token");
    if (!token) {
        console.warn("❌ No JWT token found. Redirecting to login.");
        window.location.href = "/login";
        return;
    }

    options.headers = {
        ...options.headers,
        "Authorization": `Bearer ${token}`
    };

    return fetch(url, options);
}
