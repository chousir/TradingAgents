FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages needed by some Python wheels/build steps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates nodejs npm \
    && npm install -g @google/gemini-cli \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency/install metadata first to maximize Docker layer cache reuse.
COPY requirements.txt pyproject.toml README.md ./
COPY tradingagents ./tradingagents
COPY cli ./cli
COPY main.py ./

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copy tests for container self-validation and future CI usage.
COPY tests ./tests

# Validate that core runtime deps and package imports work inside the image.
RUN command -v gemini >/dev/null \
    && gemini --help >/dev/null \
    && python -c "import tradingagents, cli, pandas, yfinance, langgraph, backtrader, typer, rich; print('Import smoke check passed')" \
    && python -m unittest \
        tests.test_model_validation \
        tests.test_ticker_symbol_handling \
        tests.test_google_api_key \
        tests.test_market_regime_tool

# Container health check: verifies package importability at runtime.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import tradingagents" || exit 1

# Default to interactive CLI; override in docker run if needed.
CMD ["tradingagents"]
