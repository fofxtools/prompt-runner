.PHONY: fmt lint check black ruff ruff-fix

fmt:
	black .

lint:
	ruff check .

ruff-fix:
	ruff check --fix .

check: fmt lint
black: fmt
ruff: lint
