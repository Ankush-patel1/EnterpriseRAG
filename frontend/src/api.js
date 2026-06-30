const BASE_URL = "http://localhost:8000";

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

