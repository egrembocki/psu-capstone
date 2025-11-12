exclude=.venv,.pytest_cache,notebooks,reports

.PHONY: help install format lint clean test update setup-dev setup-uv-windows setup-uv pre-commit env-setup

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start: 'make install'"

install: ## Install package and pre-commit hooks
	@echo "📦 Installing package in editable mode..."
	@uv sync --all-groups
	@if ! git rev-parse --git-dir >/dev/null 2>&1; then \
		echo "⚠️ Git repository not initialized. Initializing..."; \
		git init; \
		git branch -m main; \
		echo "✅ Git repository initialized with main branch"; \
	fi
	@echo "🔧 Setting up pre-commit hooks..."
	@uv run pre-commit install
	@echo "✅ Installation complete"

setup-dev: ## Setup development environment
	@echo "📚 Installing development dependencies..."
	@uv sync --all-groups
	@echo "✅ Development environment ready. Try `make test` to verify everything works"

format: ## Format code with isort and black
	@echo "🎨 Formatting code..."
	@uv run isort . --line-length=100
	@uv run black . --line-length=100
	@echo "✅ Code formatted"

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	@uv run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=$(exclude) -v
	@uv run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics --exclude=$(exclude)
	@echo "✅ Linting complete"
# stop the build if there are Python syntax errors or undefined names
# exit-zero treats all errors as warnings. The GH editor is 127 chars wide

clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf target/* dist/* build/* *.egg-info htmlcov .coverage .pytest_cache
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

update: ## Update dependencies
	@echo "🔺 Updating dependencies..."
	@uv lock --upgrade
	@echo "✅ Dependencies updated"

test: ## Run tests with coverage
	@echo "🧪 Running tests with coverage..."
	@PYTHONPATH=src/ uv run pytest \
			--cov="psu_capstone" \
			--cov-report=term-missing \
			--cov-report=html:htmlcov \
			--durations=0 \
			--disable-warnings \
			tests/
	@echo "✅ Tests complete. Coverage report: htmlcov/index.html"


setup-uv-windows: ## Install uv package manager on Windows
	@echo "🚀 Installing uv package manager..."
	@pip install uv
	@echo "✅ uv installed successfully"

setup-uv: ## Install uv package manager on Unix systems (Linux/MacOS)
	@echo "🚀 Installing uv package manager..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "✅ uv installed successfully"

pre-commit: ## Run pre-commit on all files
	@echo "🔧 Running pre-commit on all files..."
	@uv run pre-commit run --all-files
