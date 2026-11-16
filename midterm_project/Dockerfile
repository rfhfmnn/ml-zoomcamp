# Use official lightweight Python base image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=model.bin \
    OPTIMAL_THRESHOLD=0.41

# Set working directory inside container
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies and production WSGI server (Gunicorn)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt gunicorn

# Copy serialized model and prediction service code
COPY model.bin .
COPY predict.py .

# Expose standard prediction port
EXPOSE 9696

# Run production WSGI server
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "--workers=4", "predict:app"]
