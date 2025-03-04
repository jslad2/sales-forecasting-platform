document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.querySelector("#login-form");

    if (loginForm) {
        loginForm.addEventListener("submit", function (e) {
            e.preventDefault();

            const email = document.querySelector("#email").value;
            const password = document.querySelector("#password").value;

            fetch("https://ewdilyhplzrxyrbtkmjy.supabase.co/auth/v1/token?grant_type=password", {
                method: "POST",
                headers: {
                    "apikey": "your-supabase-api-key",
                    "Content-Type": "application/json"  // Ensure JSON format
                },
                body: JSON.stringify({
                    email: "jslad13@gmail.com",
                    password: "Armstr0ng1!"
                })
            })
            .then(response => response.json())
            .then(data => console.log("Login Successful:", data))
            .catch(error => console.error("Login Error:", error));
            
        });
    }
});
