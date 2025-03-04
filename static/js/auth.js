document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.querySelector("#login-form");

    if (loginForm) {
        loginForm.addEventListener("submit", function (e) {
            e.preventDefault();

            const email = document.querySelector("#email").value;
            const password = document.querySelector("#password").value;

            fetch("/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email: email, password: password })
            })
            .then(response => response.json())
            .then(data => {
                console.log("🔍 Login Response:", data);

                if (data.status === "success") {
                    console.log("✅ Redirecting to dashboard...");
                    window.location.href = data.redirect;  // Redirect on success
                } else {
                    console.error("❌ Login failed:", data.message);
                    document.querySelector("#login-error").textContent = "❌ Invalid credentials. Try again.";
                }
            })
            .catch(error => console.error("❌ Error:", error));
        });
    }
});
