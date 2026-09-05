REQUIRED_EXECUTABLES = uv rm find

.PHONY: help check test clean testpackages checkdeps

help:
	@echo ""
	@echo "  make check      Run ultra-fast static testing pipeline (ruff, bandit, vulture, etc.)"
	@echo "  make test       Run static checks followed immediately by pytest"
	@echo "  make clean      Wipe out test tool cache tracking footprints"
	@echo "  make init       Initialize new project with uv and test setup"

check:
	@echo "\n— [An extremely fast Python linter and code formatter](https://docs.astral.sh/ruff/)"
	uv run ruff check src/ tests/
	uv run ruff format src/ tests/ --check

	@echo "\n— [AST based security scanner](https://bandit.readthedocs.io/en/latest/)"
	uv run bandit -c pyproject.toml -r src/ tests/

	@echo "\n— [Find dead Python code](https://github.com/jendrikseipp/vulture)"
	uv run vulture src/ tests/ --min-confidence 80

	@echo "\n— [A tool for refurbishing and modernizing Python codebases](https://github.com/dosisod/refurb)"
	uv run refurb src/ tests/

	@echo "\n— [An extremely fast Python type checker and language server]( https://docs.astral.sh/ty/)"
	uv run ty check src/ tests/

	@echo "\n— [Interrogate a codebase for docstring coverage](https://interrogate.readthedocs.io/en/latest/)"
	#uv run interrogate src/

	@echo "\n— security scan"
	UV_MALWARE_CHECK=1 uv audit --preview-features audit-command --preview-features malware-check

test: check
	uv run pytest -v --durations=5

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .vulture_cache .uv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

init: checkdeps pyproject.toml testpackages

checkdeps:
	@$(foreach exec,$(REQUIRED_EXECUTABLES),\
		command -v $(exec) >/dev/null 2>&1 || { echo "Error: $(exec) is required."; exit 1; };)
	@echo "All required commands are available."

testpackages:
	uv add --dev ruff bandit vulture refurb ty pytest #interrogate

export GIT_CEILING_DIRECTORIES	# can influence `uv init` behaviour
pyproject.toml:
	uv init --package .
