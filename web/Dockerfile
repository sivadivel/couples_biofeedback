# Build context is the repo root (see fly.toml): docker build -f web/Dockerfile -t biofeedback-web .
FROM python:3.13-slim

WORKDIR /app

COPY web/requirements.txt web/requirements.txt
RUN pip install --no-cache-dir -r web/requirements.txt

# Shared metric engine (unmodified) + the web app itself. ble.py, simulator.py,
# transcription.py, etc. are intentionally not copied — this image never
# imports them.
COPY processor.py .
COPY specs/dyadic_biofeedback_metrics.py specs/dyadic_biofeedback_metrics.py
COPY web/ web/

ENV PORT=8080
EXPOSE 8080

CMD ["python", "web/server.py"]
