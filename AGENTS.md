# Repository Guidelines

## Project Structure & Module Organization
- `analysis/` hosts core analytics (technical tools, Monte Carlo workflows); keep new analyzers modular and well-documented.
- `visualization/` builds charts consumed by `streamlit_app.py`; reuse shared helpers to maintain dashboard consistency.
- `config/` stores runtime settings (`settings.py`, `config.yaml`); mirror new configuration keys across both files.
- `utils/` contains cross-cutting helpers; extend here rather than duplicating logic inside notebooks or apps.
- `data/` houses ingestion utilities; avoid committing raw datasets and add large files to `.gitignore`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates an isolated environment for local work.
- `pip install -r requirements.txt` installs runtime plus optional development dependencies.
- `streamlit run streamlit_app.py` launches the dashboard for manual verification.
- `pytest` executes automated checks; append `-k pattern` to target a subset during iteration.
- `black .` followed by `flake8` aligns formatting and linting with the main branch expectations.

## Coding Style & Naming Conventions
- Target Python ≥3.10, use 4-space indentation, and follow PEP 8 guidelines.
- Name modules and functions with `snake_case`, classes with `PascalCase`, and constants with `UPPER_SNAKE_CASE`.
- Add type hints and concise docstrings for public functions; group imports by stdlib, third-party, then local.

## Testing Guidelines
- Follow the existing `test_basic_functionality.py` pattern or place new suites under `tests/` using `test_*.py` names.
- Cover new logic with focused pytest functions and prefer fixtures for deterministic data setups.
- Validate edge cases (empty datasets, invalid configs) and keep stochastic routines reproducible by seeding RNGs.

## Commit & Pull Request Guidelines
- Write imperative commit messages (`Add streaming cache`, `Fix Monte Carlo seed`) and keep subjects ≤50 characters.
- Reference related issues in descriptions (`Closes #42`) and summarize behavior changes plus validation steps.
- For UI updates, attach before/after screenshots; for backend changes, share representative log or output snippets.
- Open PRs only after `pytest`, `black`, and `flake8` succeed locally; note any intentionally skipped checks with context.

## Configuration & Data Tips
- Update `config/config.yaml` and `config/settings.py` together and document new options in `README.md` or notebooks.
- Store sensitive credentials in environment variables; never commit API keys, tokens, or raw customer data.
