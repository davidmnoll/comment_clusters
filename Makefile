VENV_DIR := backend/.venv
PYTHON := python3
VENV_PYTHON := $(VENV_DIR)/bin/python

$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip

.PHONY: install build run dev fetch-comments venv

install: $(VENV_DIR)/bin/activate
	cd frontend && npm install
	cd backend && $(PYTHON) -m pip install -r requirements.txt

build-frontend:
	cd frontend && npm run build

run: $(VENV_DIR)/bin/activate frontend/dist
	cd backend && .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload


run-build: $(VENV_DIR)/bin/activate
	cd frontend && npm run build
	cd backend && .venv/bin/python cluster_comments.py
	make run

prep-backend: $(VENV_DIR)/bin/activate
	make run-fetch
	make run-cluster


dev:
	cd frontend && npm run dev

fetch-comments: $(VENV_DIR)/bin/activate
	cd backend && .venv/bin/python fetch_comments.py