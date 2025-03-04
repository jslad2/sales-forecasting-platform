document.addEventListener("DOMContentLoaded", function () {
    // Check if the URL contains an access token
    const hashParams = new URLSearchParams(window.location.hash.substring(1));
    const accessToken = hashParams.get("access_token");

    if (accessToken) {
        console.log("🔑 Access Token Found:", accessToken);

        // ✅ Send token to Flask backend to create a session
        fetch("/process-login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ token: accessToken })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "success") {
                console.log("✅ Login successful, redirecting...");
                window.location.href = "/dashboard"; // Redirect to dashboard
            } else {
                console.error("❌ Login failed:", data.error);
            }
        })
        .catch(error => console.error("❌ Error processing login:", error));
    }
});
