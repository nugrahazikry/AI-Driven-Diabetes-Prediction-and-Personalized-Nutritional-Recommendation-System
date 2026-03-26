# ── Healthkaton — Frontend (nginx) ────────────────────────────────────────────
# Build context: final/ (project root)
# Run:  docker build -f frontend/Dockerfile -t healthkaton-frontend .
# ─────────────────────────────────────────────────────────────────────────────
FROM nginx:stable-alpine

# Remove default Nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Serve static assets at /static/ inside the container
COPY frontend/public/static/ /usr/share/nginx/html/static/

# Serve main HTML entry point
COPY frontend/public/index.html /usr/share/nginx/html/index.html

# Custom Nginx config (proxies everything else to backend)
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
