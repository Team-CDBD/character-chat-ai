FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY pyproject.toml README.md ./
COPY app ./app
COPY tools ./tools
RUN pip install --no-cache-dir -e .
COPY alembic.ini ./
COPY prompts ./prompts
COPY contracts ./contracts

# ai-api by default; the worker overrides the command (see infra/docker-compose.yml)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
