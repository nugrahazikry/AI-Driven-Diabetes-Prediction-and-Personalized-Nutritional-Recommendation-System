# ── Healthkaton — Frontend (nginx) ────────────────────────────────────────────
FROM nginx:alpine

# Replace default nginx config with our custom one
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy all public assets (index.html, static/css, static/js)
COPY public/ /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
