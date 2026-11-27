FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY pyproject.toml README.md ./
COPY app ./app
COPY tools ./tools
# The tokenizer extra gives exact token counts (tiktoken for OpenAI, tokenizers for AI_TOKENIZER_FILE).
# Fetching the OpenAI encoding is best-effort: without it the service calibrates from provider usage.
ENV TIKTOKEN_CACHE_DIR=/srv/.tiktoken
RUN pip install --no-cache-dir -e '.[tokenizer]' \
 && (python -c "import tiktoken; tiktoken.get_encoding('o200k_base')" || echo "tiktoken encoding not cached")
COPY alembic.ini ./
COPY prompts ./prompts
COPY contracts ./contracts

# ai-api by default; the worker overrides the command (see infra/docker-compose.yml)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
