# ==============================================================================
# AI-Powered Trip Planner Backend - Production Dockerfile
# ==============================================================================
# This Dockerfile uses a multi-stage build to create a lean and secure
# production image.
#
# Stages:
#   1. builder: Installs dependencies and builds the application.
#   2. production: Copies the application and dependencies to a minimal
#      base image.
# ==============================================================================

# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

# Set build arguments
ARG POETRY_VERSION=1.8.2

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_CACHE_DIR=/var/cache/pypoetry

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy dependency files
WORKDIR /app
COPY poetry.lock pyproject.toml ./

# Install dependencies
RUN poetry install --no-dev --no-root

# --- Stage 2: Production ---
FROM python:3.12-slim AS production

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    ENVIRONMENT="production"

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create logs directory
RUN mkdir -p /app/logs && chown appuser:appuser /app/logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]