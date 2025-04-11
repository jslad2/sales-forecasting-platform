document.addEventListener("DOMContentLoaded", function () {
    console.log("✅ auth.js is loaded and running!");

    updateAuthButtons(); // ✅ Ensure navbar updates on page load

    const loginForm = document.querySelector("#login-form");

    if (!loginForm) {
        console.warn("❌ loginForm not found. Checking authentication...");
        checkAuth();  // ✅ Auto-check authentication if user is already logged in
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

                // ✅ Attach JWT to cookies (Optional)
                document.cookie = `access_token=${data.access_token}; path=/; Secure`;

                // ✅ Verify token BEFORE redirecting
                const tokenValid = await verifyToken();
                if (tokenValid) {
                    window.location.href = data.redirect;  // ✅ Redirect to dashboard only if valid
                } else {
                    alert("⚠️ Session expired. Please log in again.");
                    localStorage.removeItem("access_token");  // ✅ Remove invalid token
                    window.location.href = "/login";  // ✅ Redirect back to login
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
        console.warn("❌ No JWT token found.");
        
        // 🔹 Only redirect on protected pages
        const protectedPages = ["/dashboard", "/profile", "/account-settings"];
        if (protectedPages.includes(window.location.pathname)) {
            alert("⚠️ Session expired. Please log in again.");
            window.location.href = "/login";
        }
        return;
    }

    try {
        console.log("🔍 Verifying JWT token...");
        const response = await fetch("/api/verify-token", {
            method: "GET",
            headers: { "Authorization": `Bearer ${token}` }
        });

        if (!response.ok) {
            console.warn("❌ Invalid or expired token.");
            localStorage.removeItem("access_token");

            // Redirect only if the page is protected
            if (protectedPages.includes(window.location.pathname)) {
                window.location.href = "/login";
            }
        } else {
            console.log("✅ JWT verified. Access granted.");
        }
    } catch (error) {
        console.error("🔥 Error verifying token:", error);
        localStorage.removeItem("access_token");

        if (protectedPages.includes(window.location.pathname)) {
            window.location.href = "/login";
        }
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
        console.warn("❌ No JWT token found.");

        // 🔹 Prevent redirect on homepage
        if (window.location.pathname !== "/") {
            window.location.href = "/login";
        }
        return;
    }

    options.headers = {
        ...options.headers,
        "Authorization": `Bearer ${token}`
    };

    return fetch(url, options);
}

/**
 * ✅ Updates the navbar dynamically based on authentication state
 */
function updateAuthButtons() {
    console.log("🔍 Checking authentication for navbar...");
    
    const authButtons = document.getElementById("auth-buttons");
    if (!authButtons) return;

    const token = localStorage.getItem("access_token");

    if (token) {
        console.log("✅ User is logged in.");
        authButtons.innerHTML = `
            <a href="/dashboard" class="dashboard-btn">Dashboard</a>
            <a href="#" class="logout-btn" onclick="logout()">Logout</a>
        `;
    } else {
        console.log("❌ No user token found.");
        
        // 🔹 Ensure login button appears correctly
        authButtons.innerHTML = `<a href="/login" class="login-btn">Login</a>`;
    }
}

/**
 * ✅ Logout function to clear JWT and redirect to login
 */
function logout() {
    console.log("🚪 Logging out...");
    localStorage.removeItem("access_token");
    document.cookie = "access_token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 UTC"; // ✅ Remove cookie
    updateAuthButtons(); // ✅ Update navbar immediately
    window.location.href = "/";  // ✅ Redirect to homepage
}
