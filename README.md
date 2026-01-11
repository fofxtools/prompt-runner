# prompt-runner

**Under construction**

LLM evaluation toolkit for running prompts across multiple models.

## Install

```bash
pip install -e .
```

## Usage

### LLM Evaluation

```python
from prompt_runner.llm_runner import run_llm_eval
from prompt_runner.config import load_config, load_prompts, load_models

config = load_config("config/config.yaml")
prompts = load_prompts("config/llm_prompts.yaml")
models = load_models("config/llm_models.yaml")

run_llm_eval(config, prompts, models)
```

Results are saved to `storage/results/` with JSON and markdown outputs.
