# Multi-stage build for Football Analytics API
FROM python:3.12-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY mlruns/ mlruns/

# Create necessary directories
RUN mkdir -p data/bronze data/silver data/gold models mlruns

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
