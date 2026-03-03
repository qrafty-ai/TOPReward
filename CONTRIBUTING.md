# Contributing to TOPReward

Thank you for your interest in contributing to TOPReward. This guide covers how to set up your development environment, run tests, and submit changes.

## Development Setup

**Prerequisites:** Python 3.11+, [`uv`](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/TOPReward/TOPReward.git
cd TOPReward
make sync          # Install dependencies via uv
cp .env.example .env
# Fill in API keys in .env
```

## Development Workflow

```bash
make format        # Auto-format (Black, Ruff, isort)
make lint          # Lint without modifications
make test          # Run pytest suite
make pyright       # Static type checking
```

Run a single test file:
```bash
uv run pytest tests/test_voc_score.py
uv run pytest tests/test_voc_score.py::test_name -v
```

## Code Style

- Line length: **160 characters** (configured in `pyproject.toml`)
- Formatter: Black + Ruff + isort — run `make format` before committing
- Python 3.11+ type annotations are encouraged for new code


## Adding New Components

### New Model Client

1. Create `topreward/clients/my_model.py` inheriting from `BaseModelClient`
2. Implement `_generate_from_events(self, events: list[Event]) -> str`
3. Add `configs/model/my_model.yaml` with `_target_` pointing to your class
4. Use via CLI: `model=my_model`

The base class handles rate limiting, retry logic with exponential backoff, and event sequence construction.

### New Dataset

Create `configs/dataset/my_dataset.yaml` with:
- `name`, `dataset_name` (HuggingFace path or local dir)
- `camera_index`, `max_episodes`, `num_frames`, `num_context_episodes`

### New Mapper

1. Implement `topreward/mapper/my_mapper.py` inheriting from `BaseMapper`
2. Implement `extract_percentages(self, model_response: str) -> list[float]`
3. Add `configs/mapper/my_mapper.yaml`

See [README.md](README.md) for architecture and usage details.

## Submitting Changes

1. Fork the repository and create a branch from `main`
2. Make your changes and ensure all checks pass:
   ```bash
   make format && make lint && make test && make pyright
   ```
3. Clear notebook outputs before committing any `.ipynb` files:
   ```bash
   jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb
   ```
4. Open a pull request against `main` with a clear description of what changed and why

## Reporting Issues

Please open a GitHub issue with:
- A minimal reproducible example
- The full error traceback
- Your Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
