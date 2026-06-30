const BASE_URL = "http://localhost:8000";

export async function askQuestion(question) {
  const response = await fetch(`${BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    throw new Error("Request failed");
  }

  return response.json();
}
