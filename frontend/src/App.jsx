import { useState } from "react";
import { askQuestion } from "./api";
import "./App.css";

// Parses answer text and wraps [citation] markers in highlighted spans
function AnswerWithCitations({ text }) {
  const parts = text.split(/(\[[^\]]+\])/g);
  return (
    <p className="answer-text">
      {parts.map((part, i) =>
        /^\[.+\]$/.test(part) ? (
          <mark key={i} className="citation-mark">{part}</mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </p>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleAsk() {
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setAnswer(null);
    setSources([]);

    try {
      const data = await askQuestion(question);
      setAnswer(data.answer);
      setSources(data.sources);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter") handleAsk();
  }

  return (
    <div className="page">
      <header className="header">
        <h1 className="title">Enterprise Knowledge Assistant</h1>
        <p className="subtitle">Ask questions about enterprise documents.</p>
      </header>

      <main className="main">
        <div className="search-box">
          <input
            className="input"
            type="text"
            placeholder="Type your question here..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            className="btn"
            onClick={handleAsk}
            disabled={loading || !question.trim()}
          >
            {loading ? "Searching..." : "Ask"}
          </button>
        </div>

        {error && <p className="error">{error}</p>}

        {answer && (
          <div className="result-card">

            <div className="result-section">
              <p className="label">Question</p>
              <p className="question-text">{question}</p>
            </div>

            <div className="result-section">
              <p className="label">Answer</p>
              <AnswerWithCitations text={answer} />
            </div>

            {sources.length > 0 && (
              <div className="result-section">
                <p className="label">Sources</p>
                <ul className="sources-list">
                  {sources.map((source, index) => (
                    <li key={index} className="source-item">
                      <span className="source-filename">{source.filename}</span>
                      <span className="source-page">Page {source.page}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
