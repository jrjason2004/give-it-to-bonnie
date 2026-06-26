FROM python:3.12-slim

# ffmpeg + ffprobe (intro audio mix, video compositing, wav duration)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Render injects $PORT; bind on all interfaces so the platform can route to us
ENV HOST=0.0.0.0
EXPOSE 10000
CMD ["python", "landing.py"]
