FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY jobpilot ./jobpilot
COPY static ./static
COPY scripts ./scripts
COPY run.py .

# Don't run as root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
# run.py reads $PORT (Render sets it) and binds 0.0.0.0.
CMD ["python", "run.py"]
