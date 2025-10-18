# Repository Guidelines

## Project Structure & Module Organization
Fairseq's core library lives in `fairseq/`, grouped by domain (models, criterions, tasks, optim, config). Command-line entry points and Hydra defaults are in `fairseq_cli/` and `fairseq/config/`. Research recipes reside under `examples/`, while reusable utilities sit in `scripts/`. Tests live in `tests/` with subdirectories for distributed, GPU, and speech workflows; mirror this layout when adding coverage. Documentation sources are in `docs/`, and optional Hydra plugins are packaged under `hydra_plugins/`.

## Build, Test, and Development Commands
Install dependencies with `pip install -e .[dev]` to pull in pytest, flake8, and black. Rebuild C extensions, when necessary, with `python setup.py build_ext --inplace`. Run the full suite via `pytest tests`, or target a module with `pytest tests/test_sequence_generator.py -k beam`. Execute CLI entry points using installed scripts (e.g., `fairseq-train --task translation --config-dir examples/translation/conf`).

## Coding Style & Naming Conventions
Code is formatted with black (88-char lines) and import-sorted with isort; install the repo's pre-commit hooks to enforce these automatically. Follow standard Python 4-space indentation, prefer `snake_case` for functions and variables, and reserve `CamelCase` for classes and configs. Modules registering with Fairseq's registry should call `@register_*` decorators near the definition to keep discovery predictable. Favor type hints for new public APIs.

## Testing Guidelines
Use pytest for unit and integration tests; mirror existing naming (`test_<feature>.py`) and group helpers in `tests/utils.py`. GPU-only cases should be guarded with `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`, and distributed scenarios belong in `tests/distributed/`. Include deterministic seeds in new tests to keep CI reproducible.

## Commit & Pull Request Guidelines
Commits in this repo use concise, present-tense summaries (e.g., “Add warning for potential security vulnerabilities...”) and often append the tracking PR number. Squash noisy fixup commits before review. When opening a pull request, describe the motivation, list user-visible changes, and paste the exact test command output. Link relevant issues, attach config snippets for new recipes, and add screenshots only when touching docs or UIs.

## Configuration & Security Tips
Hydra drives configuration; prefer updating YAML under `fairseq/config/` and reference it via `--config-dir` instead of duplicating flags. Defaults assume trusted networks—when launching distributed training, set explicit `--distributed-world-size` and firewall rendezvous endpoints to avoid the distributed-mode warning introduced in #5635.
