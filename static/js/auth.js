document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.querySelector("#login-form");

    if (loginForm) {
        loginForm.addEventListener("submit", async function (e) {
            e.preventDefault();

            const email = document.querySelector("#email").value.trim();
            const password = document.querySelector("#password").value.trim();

            if (!email || !password) {
                alert("⚠️ Please enter both email and password.");
                return;
            }

            try {
                console.log("🔍 Sending login request...");

                const response = await fetch("/login", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ email, password }),
                });

                // ✅ Handle non-JSON responses
                const contentType = response.headers.get("content-type");
                if (!contentType || !contentType.includes("application/json")) {
                    throw new Error("Received non-JSON response from server.");
                }

                const data = await response.json();
                console.log("✅ Login Response:", data);

                if (response.ok && data.status === "success") {
                    alert("✅ Login successful! Redirecting...");
                    window.location.href = data.redirect;  // Redirect to dashboard
                } else {
                    alert(`❌ Login failed: ${data.message || "Invalid credentials"}`);
                }
            } catch (error) {
                console.error("🔥 Login Error:", error);
                alert("❌ An error occurred. Please try again.");
            }
        });
    }
});
