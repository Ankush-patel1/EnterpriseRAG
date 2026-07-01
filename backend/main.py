import uvicorn

if __name__ == "__main__":
    # Local development only.
    # On Render, Gunicorn starts the app via the Dockerfile CMD.
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
