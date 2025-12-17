# Dockerfile for Cloud Run deployment
# Includes Chromium and all dependencies for browser-use and Daily streaming

FROM python:3.11-slim

# Install system dependencies including Chromium
RUN apt-get update && apt-get install -y \
    # Chromium and dependencies
    chromium \
    chromium-driver \
    # Additional Chrome dependencies
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libcairo2 \
    # Other utilities
    wget \
    curl \
    ca-certificates \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for headless Chrome
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMIUM_PATH=/usr/bin/chromium
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY daily_streaming/requirements.txt daily_streaming/requirements.txt

# Install Python dependencies
# Note: daily-python is available on Linux
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r daily_streaming/requirements.txt && \
    pip install --no-cache-dir daily-python

# Install browser-use browsers (Chromium via Playwright)
RUN python -m playwright install chromium --with-deps

# Copy application code
COPY fastapi_agent.py .
COPY daily_streaming/ daily_streaming/
COPY .env* ./ 

# Create directory for browser data 
RUN mkdir -p /tmp/browser-data && chmod 777 /tmp/browser-data

# Expose port (Cloud Run uses PORT env var, default 8080)
ENV PORT=8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Run with uvicorn (single worker for WebSocket support)
CMD uvicorn fastapi_agent:app --host 0.0.0.0 --port ${PORT} --workers 1
