document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.querySelector("#login-form");

    if (loginForm) {
        loginForm.addEventListener("submit", function (e) {
            e.preventDefault();  // Prevent form from reloading the page

            const email = document.querySelector("#email").value;
            const password = document.querySelector("#password").value;

            fetch("/login", {  // ✅ Use your Flask backend API
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ email, password })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === "success") {
                    console.log("✅ Login Successful:", data);
                    window.location.href = data.redirect;  // ✅ Redirect to dashboard
                } else {
                    console.error("❌ Login Failed:", data.message);
                    document.querySelector("#error-message").innerText = data.message;  // Show error to user
                }
            })
            .catch(error => console.error("🔥 Login Error:", error));
        });
    }
});
