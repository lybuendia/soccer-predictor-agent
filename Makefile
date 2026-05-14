VENV_PYTHON := ./.venv/bin/python
VENV_PIP := ./.venv/bin/pip

.PHONY: venv install test test-live

venv:
	@test -x "$(VENV_PYTHON)" || python3.11 -m venv .venv

install: venv
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	$(VENV_PIP) install -e .

test: venv
	$(VENV_PYTHON) -m pytest -q -m "not live_llm"

test-live: venv
	$(VENV_PYTHON) -m pytest -q
