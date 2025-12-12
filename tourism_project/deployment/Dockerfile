# tourism_project/deployment/Dockerfile
# Small, production-friendly Dockerfile for the Streamlit deployment.

# Use a slim Python base to keep image smaller
FROM python:3.9-slim

# Prevent Python from buffering stdout/stderr (useful for logs)
ENV PYTHONUNBUFFERED=1

# Create a non-root user early for security
RUN adduser --disabled-password --gecos "" --uid 1000 appuser

# Set working dir
WORKDIR /app

# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt /app/requirements.txt

# Install dependencies (no cache to reduce image size)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code and set ownership to the non-root user
# using --chown (supported by modern Docker)
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Expose Streamlit default port (for documentation only)
EXPOSE 8501

# Default command to run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
