document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.querySelector("#login-form");

    if (loginForm) {
        loginForm.addEventListener("submit", async function (e) {
            e.preventDefault();  // ✅ Prevent form from default GET submission

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

                let data;
                try {
                    data = await response.json();
                } catch (error) {
                    console.error("Failed to parse JSON response:", error);
                    alert("❌ An unexpected error occurred. Please try again.");
                    return;
                }

                if (response.ok && data.status === "success") {
                    alert("✅ Login successful! Redirecting...");
                    window.location.href = data.redirect;
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
    }
});