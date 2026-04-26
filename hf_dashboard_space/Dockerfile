FROM nginx:1.27-alpine

# Hugging Face Docker Spaces expect the app on port 7860.
# We configure nginx to listen on 7860 and serve a static site at /.

COPY hf_dashboard_space/nginx.conf /etc/nginx/conf.d/default.conf

# Copy the static dashboard (HTML/CSS/JS + data.js + plots/)
COPY dashboard/site/ /usr/share/nginx/html/

EXPOSE 7860

