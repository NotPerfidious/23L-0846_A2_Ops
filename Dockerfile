FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

RUN useradd -m appuser

WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code
COPY . .

# Transfer ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Adding a health check hitting the /health route created in Task 4.2
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1


EXPOSE 5000

CMD ["python", "app.py"]