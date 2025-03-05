document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.querySelector("#login-form");

    if (loginForm) {
        loginForm.addEventListener("submit", async function (e) {
            e.preventDefault();

            const email = document.querySelector("#email").value.trim();
            const password = document.querySelector("#password").value.trim();
            const loginButton = document.querySelector("#login-button");

            if (!email || !password) {
                alert("⚠️ Please enter both email and password.");
                return;
            }

            if (loginButton) {
                loginButton.disabled = true;  // ✅ Disable button to prevent multiple requests
                loginButton.textContent = "Logging in...";  // ✅ Update UI feedback
            }

            try {
                console.log("🔍 Sending login request...");

                const response = await fetch("/api/login", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ email, password })
                });

                // ✅ Handle non-JSON responses
                const contentType = response.headers.get("content-type");
                if (!contentType || !contentType.includes("application/json")) {
                    throw new Error("Received non-JSON response from server.");
                }

                const data = await response.json();
                console.log("✅ Login Response:", data);

                // ✅ Handle 405 Method Not Allowed
                if (response.status === 405) {
                    throw new Error("Method Not Allowed: Check if the API endpoint accepts POST requests.");
                }

                // ✅ Handle other server errors
                if (!response.ok) {
                    throw new Error(`Server Error: ${data.message || response.statusText}`);
                }

                if (data.status === "success") {
                    alert("✅ Login successful! Redirecting...");

                    // ✅ Store access token & user tier for authentication persistence
                    localStorage.setItem("access_token", data.access_token);
                    localStorage.setItem("user_tier", data.tier);

                    window.location.href = data.redirect;  // ✅ Redirect to dashboard
                } else {
                    alert(`❌ Login failed: ${data.message || "Invalid credentials"}`);
                }
            } catch (error) {
                console.error("🔥 Login Error:", error);
                alert(`❌ Error: ${error.message}`);
            } finally {
                if (loginButton) {
                    loginButton.disabled = false;  // ✅ Re-enable button
                    loginButton.textContent = "Login";  // ✅ Reset text
                }
            }
        });
    }
});
