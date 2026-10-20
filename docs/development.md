# Development tools

## Formatting (similar to PHP-CS-Fixer)
```
black src/ tests/
```

## Code smells (similar to PHP CodeSniffer)
Check only (safe):

```
ruff check src/ tests/
```

You can auto-fix with `ruff check --fix src/ tests/`. However auto-fix can make behavioral changes. Use with care.

## Correctness (similar to PHPStan)
Run on folders separately to avoid module source file conflicts:

```
mypy src/
mypy tests/
```

## Testing (similar to PHPUnit)
```
pytest         # everything except the GPU test
pytest -m gpu  # only the GPU smoke test (needs a free GPU, sd-server and the Klein 4B files)
pytest -m all  # everything, GPU test included
```