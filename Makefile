IMAGE_NAME ?= tradingagents
IMAGE_TAG ?= $(shell date +%Y%m%d)
IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)
CONTAINER_NAME ?= tradingagents-app
ENV_FILE ?= .env

REPORTS_DIR ?= reports
RESULTS_DIR ?= results
GEMINI_CONFIG_DIR ?= $(HOME)/.gemini

ENV_ARG := $(if $(wildcard $(ENV_FILE)),--env-file $(ENV_FILE),)

.DEFAULT_GOAL := docker-build

.PHONY: help docker-build docker-run docker-shell docker-gemini-login docker-test docker-smoke docker-stop docker-rm docker-rmi docker-logs

help:
	@echo "Available targets:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run TradingAgents CLI container (interactive)"
	@echo "  make docker-shell   - Open shell inside container image"
	@echo "  make docker-gemini-login - Login Gemini CLI with Google account (persisted)"
	@echo "  make docker-test    - Run unit tests inside container"
	@echo "  make docker-smoke   - Run import smoke checks inside container"
	@echo "  make docker-stop    - Stop running container"
	@echo "  make docker-rm      - Remove container"
	@echo "  make docker-rmi     - Remove image"
	@echo "  make docker-logs    - Show container logs"

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		$(ENV_ARG) \
		-v $(GEMINI_CONFIG_DIR):/root/.gemini \
		-v $(CURDIR)/$(REPORTS_DIR):/app/$(REPORTS_DIR) \
		-v $(CURDIR)/$(RESULTS_DIR):/app/$(RESULTS_DIR) \
		$(IMAGE)

docker-shell:
	docker run --rm -it \
		--name $(CONTAINER_NAME)-shell \
		$(ENV_ARG) \
		-v $(GEMINI_CONFIG_DIR):/root/.gemini \
		-v $(CURDIR):/app \
		-w /app \
		$(IMAGE) /bin/bash

docker-gemini-login:
	mkdir -p $(GEMINI_CONFIG_DIR)
	docker run --rm -it \
		--name $(CONTAINER_NAME)-gemini-login \
		-v $(GEMINI_CONFIG_DIR):/root/.gemini \
		$(IMAGE) gemini

docker-test:
	docker run --rm \
		--name $(CONTAINER_NAME)-test \
		$(ENV_ARG) \
		-v $(CURDIR):/app \
		-w /app \
		$(IMAGE) \
		python -m unittest \
			tests.test_model_validation \
			tests.test_ticker_symbol_handling \
			tests.test_google_api_key \
			tests.test_market_regime_tool

docker-smoke:
	docker run --rm \
		--name $(CONTAINER_NAME)-smoke \
		$(ENV_ARG) \
		$(IMAGE) \
		python -c "import tradingagents, cli, pandas, yfinance, langgraph, backtrader, typer, rich; print('Import smoke check passed')"

docker-stop:
	-docker stop $(CONTAINER_NAME)

docker-rm:
	-docker rm $(CONTAINER_NAME)

docker-rmi:
	docker rmi $(IMAGE)

docker-logs:
	docker logs -f $(CONTAINER_NAME)
