document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.querySelector("#login-form");

    if (loginForm) {
        loginForm.addEventListener("submit", function (e) {
            e.preventDefault();

            const email = document.querySelector("#email").value.trim();
            const password = document.querySelector("#password").value.trim();

            fetch("/login", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ email, password })
            })
            .then(response => response.json())
            .then(data => {
                console.log("🔍 Login Response:", data);
                
                if (data.status === "success") {
                    // ✅ Store JWT token in localStorage
                    localStorage.setItem("access_token", data.access_token);
                    
                    // ✅ Redirect to dashboard
                    window.location.href = data.redirect;
                } else {
                    showErrorMessage(data.message || "Login failed! Please try again.");
                }
            })
            .catch(error => {
                console.error("🔥 Login Error:", error);
                showErrorMessage("An unexpected error occurred. Please try again.");
            });
        });
    }
});

/**
 * ✅ Display an error message properly
 * @param {string} message - The error message to display
 */
function showErrorMessage(message) {
    const errorContainer = document.querySelector("#error-message");
    if (errorContainer) {
        errorContainer.textContent = message;
        errorContainer.style.display = "block";
    } else {
        alert(message);  // Fallback if no error container exists
    }
}
