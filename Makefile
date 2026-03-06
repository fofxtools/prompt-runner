.PHONY: fmt lint check black ruff

fmt:
	black src/ tests/ examples/ scripts/

lint:
	ruff check src/ tests/ examples/ scripts/

check: fmt lint
black: fmt
ruff: lint
