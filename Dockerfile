# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    NLTK_DATA=/usr/local/share/nltk_data

# libgomp1 is required by LightGBM at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first to maximise layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the NLTK resources used by app.py into the image
RUN python -m nltk.downloader -d ${NLTK_DATA} stopwords wordnet omw-1.4

# Copy only what the API needs at runtime
COPY app.py .
COPY tfidf_vectorizer.pkl .

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

# MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD are supplied at runtime
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]