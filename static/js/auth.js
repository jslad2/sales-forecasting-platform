document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.querySelector("#login-form");

    if (loginForm) {
        loginForm.addEventListener("submit", async function (e) {
            e.preventDefault();

            const email = document.querySelector("#email").value.trim();
            const password = document.querySelector("#password").value.trim();
            const loginButton = document.querySelector("#login-button"); // Assume you have a login button

            if (!email || !password) {
                alert("⚠️ Please enter both email and password.");
                return;
            }

            loginButton.disabled = true;  // ✅ Disable button to prevent multiple requests
            loginButton.textContent = "Logging in...";  // ✅ Update UI feedback

            try {
                console.log("🔍 Sending login request...");

                const response = await fetch("/login", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    body: JSON.stringify({ email, password })
                });

                const contentType = response.headers.get("content-type");
                if (!contentType || !contentType.includes("application/json")) {
                    throw new Error("Received non-JSON response from server.");
                }

                const data = await response.json();
                console.log("✅ Login Response:", data);

                if (response.ok && data.status === "success") {
                    alert("✅ Login successful! Redirecting...");

                    // ✅ Store access token in localStorage for authentication persistence
                    localStorage.setItem("access_token", data.access_token);
                    localStorage.setItem("user_tier", data.tier);  // Store user tier for feature access

                    window.location.href = data.redirect;  // ✅ Redirect to dashboard
                } else {
                    alert(`❌ Login failed: ${data.message || "Invalid credentials"}`);
                }
            } catch (error) {
                console.error("🔥 Login Error:", error);
                alert("❌ An error occurred. Please check your internet connection and try again.");
            } finally {
                loginButton.disabled = false;  // ✅ Re-enable button
                loginButton.textContent = "Login";  // ✅ Reset text
            }
        });
    }
});
