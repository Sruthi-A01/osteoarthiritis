# -----------------------------
# Base image (Python 3.10 – ML safe)
# -----------------------------
FROM python:3.10-slim

# Disable .pyc files and enable logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies needed for ML & matplotlib
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create runtime folders
RUN mkdir -p uploads app/static/plots

# Expose Flask port
EXPOSE 6055

# -----------------------------
# Run Flask application
# -----------------------------
CMD ["python", "main.py"]

