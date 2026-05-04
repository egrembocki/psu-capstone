# Contributing and development setup

This document complements the main [README](README.md) with environment details, Make, and git conventions.

## 1. Environment: uv

This project uses [uv](https://github.com/astral-sh/uv) for reproducible Python environments. **Python 3.12+** is required (`requires-python` in `pyproject.toml`).

### Install uv

- **Windows (PowerShell):**

  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```

- **Linux / macOS:**

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Verify:**

  ```bash
  uv --version
  ```

### Create and sync the environment

From the repository root:

```bash
uv python install 3.12
uv python pin 3.12
uv sync --all-groups
```

uv keeps the virtual environment in `.venv`; commands such as `uv run` use it automatically.

### Optional: recreate the environment

Use this when `.venv` points to a removed or incompatible Python, or dependency resolution is broken.

1. **Makefile (recommended):**

   ```bash
   make recreate-venv
   ```

   Removes `.venv`, installs and pins Python 3.12, and runs `uv sync --all-groups`.

2. **Manually (Unix / macOS):**

   ```bash
   rm -rf .venv
   uv python install 3.12
   uv python pin 3.12
   uv sync --all-groups
   ```

3. **Manually (Windows PowerShell):**

   ```powershell
   Remove-Item -Recurse -Force .venv
   uv python install 3.12
   uv python pin 3.12
   uv sync --all-groups
   ```

After recreation, you can refresh locks:

```bash
uv lock --upgrade
uv sync --all-groups
```

## 2. Make

The [Makefile](Makefile) automates install, tests, lint, and format tasks.

### Install Make

- **Linux (Debian / Ubuntu):** `sudo apt-get update && sudo apt-get install build-essential`
- **macOS:** `xcode-select --install`
- **Windows:** use WSL or Git Bash, or `choco install make` with [Chocolatey](https://chocolatey.org/)

### Makefile dependencies

- `uv` (see above)
- Dev tools (`pre-commit`, `isort`, `black`, `flake8`, `pytest`, …) come from `uv sync --all-groups`

From the repo root, run `make help` for all targets.

## 3. Git workflow

1. **Stage:** `git add .`
2. **Commit:** `git commit -m "Your message"` — if pre-commit fails, fix issues, re-stage, and commit again.
3. **Sync:** `git pull --rebase` then `git push`

## 4. Notes

- `uv run` executes commands inside the managed virtual environment.
- Prefer not editing `.venv` by hand; use `make recreate-venv` when needed.
