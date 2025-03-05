console.log("✅ auth.js is loaded and running!"); // Debugging log

document.addEventListener("DOMContentLoaded", function () {
    console.log("✅ DOM fully loaded, initializing login script...");

    const loginForm = document.querySelector("#login-form");

    if (loginForm) {
        console.log("✅ Found login form! Adding event listener...");

        loginForm.addEventListener("submit", async function (e) {
            e.preventDefault();
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
                    localStorage.setItem("access_token", data.access_token);
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
    } else {
        console.log("⚠️ loginForm not found! JavaScript not running?");
    }
});
