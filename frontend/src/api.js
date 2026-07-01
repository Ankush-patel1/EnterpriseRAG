// VITE_API_URL is set as an environment variable in the Render dashboard.
// Falls back to localhost:8000 for local development.
const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function askQuestion(question) {
  const response = await fetch(`${BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    let detail = "Request failed";
    try {
      const json = await response.json();
      detail = json.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }

  return response.json();
}


