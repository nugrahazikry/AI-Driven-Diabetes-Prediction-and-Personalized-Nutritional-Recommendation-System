# ── Healthkaton — Backend (Flask API) ────────────────────────────────────────
FROM python:3.11-slim

# System deps for Pillow / numpy + curl for Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application source
COPY . .

EXPOSE 5000

# 2 workers, 120 s timeout (AI calls can be slow)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app:app"]
