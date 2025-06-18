# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy consolidated requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directory for data
RUN mkdir -p ./data

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Set environment variables
ENV EXCEL_PATH=./data/Sentiment&Attributes_Classification.xlsx
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8501

# Health check (Streamlit does not have a direct health endpoint like FastAPI, 
# so we'll rely on the process running for now, or add a custom one if needed)
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8501 || exit 1

# Run the startup script
CMD ["./start.sh"]


