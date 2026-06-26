# Repo-root Dockerfile so Render finds it with the default (repo-root) build context.
# The app lives in production/; we copy that in.
FROM python:3.12-slim

# ffmpeg + ffprobe (intro audio mix, compositing)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY production/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY production/ .

# Render injects $PORT; bind on all interfaces
ENV HOST=0.0.0.0
EXPOSE 10000
CMD ["python", "landing.py"]
