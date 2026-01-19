# Makefile for Cloud Service Integration with MCP
# ================================================

.PHONY: help install dev test lint format clean docker-up docker-down docker-build migrate

# Default target
help:
	@echo "Cloud Service Integration with MCP - Development Commands"
	@echo "========================================================="
	@echo ""
	@echo "Development:"
	@echo "  make install      - Install dependencies"
	@echo "  make dev          - Start development environment"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-e2e     - Run e2e tests only"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build all Docker images"
	@echo "  make docker-up    - Start all services"
	@echo "  make docker-down  - Stop all services"
	@echo "  make docker-logs  - View service logs"
	@echo ""
	@echo "Database:"
	@echo "  make migrate      - Run database migrations"
	@echo "  make db-reset     - Reset database"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove build artifacts"

# ===================
# Development
# ===================

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

dev:
	docker-compose up -d postgres redis localstack
	@echo "Waiting for services..."
	@sleep 5
	python -m phase_2_langgraph_orchestrator.workflow

test:
	pytest tests/ -v --cov=. --cov-report=html

test-unit:
	pytest tests/unit/ -v -m "not integration and not e2e"

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

lint:
	ruff check .
	mypy phase_1_mcp_core phase_2_langgraph_orchestrator shared --ignore-missing-imports

format:
	black .
	isort .
	ruff check --fix .

# ===================
# Docker
# ===================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Services starting..."
	@echo "  - Orchestrator API: http://localhost:8080"
	@echo "  - Grafana:          http://localhost:3000 (admin/admin)"
	@echo "  - Prometheus:       http://localhost:9090"
	@echo "  - Jaeger:           http://localhost:16686"
	@echo "  - OPA:              http://localhost:8181"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v --rmi local
	docker system prune -f

# ===================
# Database
# ===================

migrate:
	docker-compose exec postgres psql -U postgres -d orchestrator -f /docker-entrypoint-initdb.d/init.sql

db-reset:
	docker-compose down -v postgres
	docker-compose up -d postgres
	@sleep 5
	$(MAKE) migrate

db-shell:
	docker-compose exec postgres psql -U postgres -d orchestrator

# ===================
# MCP Servers
# ===================

run-mcp-aws:
	python -m phase_1_mcp_core.servers.aws.aws_server

run-mcp-azure:
	python -m phase_1_mcp_core.servers.azure.azure_server

run-mcp-gcp:
	python -m phase_1_mcp_core.servers.gcp.gcp_server

# ===================
# Orchestrator
# ===================

run-orchestrator:
	python -m phase_2_langgraph_orchestrator.workflow

run-api:
	uvicorn phase_2_langgraph_orchestrator.api:app --host 0.0.0.0 --port 8080 --reload

# ===================
# Cleanup
# ===================

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name "*.egg-info" -delete
	rm -rf build dist htmlcov .coverage

# ===================
# LocalStack
# ===================

localstack-setup:
	@echo "Setting up LocalStack resources..."
	aws --endpoint-url=http://localhost:4566 s3 mb s3://test-bucket
	aws --endpoint-url=http://localhost:4566 sqs create-queue --queue-name test-queue
	@echo "LocalStack resources created"

# ===================
# Security
# ===================

security-scan:
	bandit -r phase_1_mcp_core phase_2_langgraph_orchestrator shared -ll
	safety check -r requirements.txt

# ===================
# Documentation
# ===================

docs:
	cd docs && mkdocs build

docs-serve:
	cd docs && mkdocs serve
