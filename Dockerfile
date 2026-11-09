FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home --uid 10001 netrag
COPY netrag/ ./netrag/
USER netrag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "netrag.app:app", "--host", "0.0.0.0", "--port", "8000"]
