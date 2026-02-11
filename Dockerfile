FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.0.0

COPY pyproject.toml poetry.lock* README.md /app/

RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --only main --no-root

COPY . /app

EXPOSE 7860

CMD ["poetry", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
