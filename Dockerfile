FROM python:3.11-slim

# libgl / libglib are required by opencv even in the headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY ml/ ./ml/
COPY models/ ./models/
COPY samples/ ./samples/
COPY healthcheck.py ./healthcheck.py

# Build the reference gallery at image-build time so the demo works offline.
RUN python ml/make_samples.py

ENV PYTHONPATH=/app/backend
ENV PYTHONUNBUFFERED=1

# Hosts disagree about which port to use and how to tell the app: Hugging Face
# Spaces expects 7860, Render and Google Cloud Run inject $PORT at runtime, and
# Fly.io reads fly.toml. Defaulting to 7860 while honouring $PORT satisfies all
# of them without a per-host Dockerfile.
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD ["python", "/app/healthcheck.py"]

# Shell form so ${PORT} is expanded at container start rather than build time.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --app-dir backend
