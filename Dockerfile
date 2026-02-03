# ALIN Framework Docker Container
# ================================
# Reproducible environment for running the ALIN pipeline
#
# Build:
#   docker build -t alin-framework .
#
# Run:
#   docker run -v $(pwd)/depmap_data:/app/depmap_data -v $(pwd)/results:/app/results alin-framework
#
# Interactive:
#   docker run -it -v $(pwd)/depmap_data:/app/depmap_data alin-framework bash

FROM python:3.12-slim

LABEL maintainer="Roy Erzurumluoğlu <roy.erzurumluoglu@gmail.com>"
LABEL description="ALIN Framework - Adaptive Lethal Intersection Network for drug combination discovery"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt requirements-lock.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-lock.txt

# Copy source code
COPY . .

# Create directories for data and results
RUN mkdir -p depmap_data results validation_data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command: run full pipeline
CMD ["python", "pan_cancer_xnode.py", "--all-cancers", "--output", "results/"]

# Alternative entry points:
# Run for specific cancer:
#   docker run alin-framework python pan_cancer_xnode.py --cancer-type "Pancreatic Adenocarcinoma"
#
# Run tests:
#   docker run alin-framework pytest tests/ -v
#
# Interactive Python:
#   docker run -it alin-framework python
