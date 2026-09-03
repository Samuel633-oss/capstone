
FROM python:3.11-slim

WORKDIR /app


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


COPY requirements.txt .


RUN pip install --no-cache-dir \
    -r requirements.txt


COPY . .

# data/account_history_live.json now persists transaction history across
# process restarts, but a container filesystem itself is still ephemeral
# on redeploy. Mount a volume at /app/data in production so history
# survives redeploys too, e.g.: docker run -v gotcha_data:/app/data ...
VOLUME ["/app/data"]

EXPOSE 8080


CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
