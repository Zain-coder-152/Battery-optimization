# ---------- 1. build the React dashboard ----------
FROM node:24-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- 2. Python API that also serves the dashboard ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api.py core.py agent.py ./
COPY data/locations.json data/
COPY data/raw/solar_*.csv data/raw/
COPY data/processed/hourly_dataset.csv data/processed/
COPY --from=web /web/dist frontend/dist
RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"
# hosting platforms (Render, Railway, Fly) pass the port to listen on in $PORT
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port $PORT"]
