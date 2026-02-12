FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Nginx pour reverse proxy (un seul port public: 7860)
RUN apt-get update \
 && apt-get install -y --no-install-recommends nginx \
 && rm -rf /var/lib/apt/lists/*

# Poetry
RUN pip install --no-cache-dir poetry==2.0.0

COPY pyproject.toml poetry.lock* README.md /app/

RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --only main --no-root

COPY . /app

# Nginx conf + start script
COPY nginx.conf /etc/nginx/nginx.conf
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]
