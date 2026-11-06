"""Entry point for Vercel: it runs the FastAPI app (API + built dashboard) as one serverless function.
Locally, keep using `uvicorn api:app`."""
from api import app  # noqa: F401
