FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    zip \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py .
COPY featureExtraction.py .
COPY librerias.py .
COPY main.py .
COPY modelo.py .
COPY textProcessing.py .

# Create necessary directories
RUN mkdir -p ./data/input ./data/output ./data/modelos ./mlruns
RUN chmod -R 777 ./data ./mlruns

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV PYTHONPATH=/app

# Copy all files from data/input directory
COPY ./data/input/  ./data/input/

# Copy entrypoint script
COPY entrypoint.sh .
# Fix line endings and make executable
RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh

# Expose ports for MLflow and Prefect
EXPOSE 5000 4200

# Run entrypoint script
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]