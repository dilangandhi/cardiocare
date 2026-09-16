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

# Build the reference gallery at image-build time so the demo works offline.
RUN python ml/make_samples.py

ENV PYTHONPATH=/app/backend
ENV PYTHONUNBUFFERED=1

# 7860 is the port Hugging Face Spaces expects.
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:7860/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "backend"]
