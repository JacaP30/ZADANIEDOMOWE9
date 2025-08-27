# Dockerfile dla FastAPI backend
FROM python:3.11-slim

WORKDIR /app

# Kopiuj requirements i zainstaluj zależności
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiuj kod aplikacji
COPY . .

# Kopiuj build React
COPY ../build ./build

# Expose port
EXPOSE 8000

# Komenda uruchomienia
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
