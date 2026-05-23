PYTHON ?= python3.11
VENV ?= venv
PIP := $(VENV)/bin/pip

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

check:
	$(PYTHON) -m compileall src

frz:
	$(PIP) freeze > requirements.txt
