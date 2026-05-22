PYTHON ?= python3.11
VENV ?= venv
PIP := $(VENV)/bin/pip
JUPYTER := $(VENV)/bin/jupyter

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

notebook:
	$(JUPYTER) notebook main.ipynb

check:
	$(PYTHON) -m compileall src

frz:
	$(PIP) freeze > requirements.txt
