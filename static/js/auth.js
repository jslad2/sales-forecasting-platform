document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.querySelector("#login-form");

    if (loginForm) {
        loginForm.addEventListener("submit", function (e) {
            e.preventDefault();

            const email = document.querySelector("#email").value;
            const password = document.querySelector("#password").value;

            fetch("/login", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",  // ✅ Ensure JSON content type
                },
                body: JSON.stringify({ email, password })  // ✅ Ensure JSON format
            })
            .then(response => response.json())
            .then(data => {
                console.log("Login Response:", data);
                
                if (data.status === "success") {
                    window.location.href = data.redirect;  // ✅ Redirect to dashboard
                } else {
                    alert(data.message || "Login failed! Please try again.");
                }
            })
            .catch(error => {
                console.error("Login Error:", error);
                alert("An error occurred. Please try again.");
            });
        });
    }
});
