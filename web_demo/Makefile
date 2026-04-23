.PHONY: help install install-web install-inference dev-web dev-inference infer health clean

INFERENCE_DIR := services/inference
WEB_DIR       := apps/web
PORT          ?= 8000
PYTHON        ?= $(shell command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3.10 2>/dev/null || echo python3)
VENV          := $(INFERENCE_DIR)/.venv
VPY           := $(VENV)/bin/python
VPIP          := $(VENV)/bin/pip

help:
	@echo "Targets:"
	@echo "  install        Install JS + Python deps"
	@echo "  dev-inference  Run FastAPI inference service on :$(PORT)"
	@echo "  dev-web        Run Next.js dev server"
	@echo "  infer IMAGE=... MASK=...  Run the CLI on a single image"
	@echo "  health         curl the /health endpoint"

install: install-web install-inference

install-web:
	cd $(WEB_DIR) && npm install

install-inference: $(VENV)/.installed

$(VENV)/.installed: $(INFERENCE_DIR)/requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VPIP) install --upgrade pip
	$(VPIP) install -r $(INFERENCE_DIR)/requirements.txt
	$(VPIP) install -e $(INFERENCE_DIR)
	touch $@

dev-inference: install-inference
	cd $(INFERENCE_DIR) && ../../$(VENV)/bin/uvicorn inference_service.main:app --host 0.0.0.0 --port $(PORT) --reload

dev-web:
	cd $(WEB_DIR) && npm run dev

infer: install-inference
	cd $(INFERENCE_DIR) && ../../$(VENV)/bin/inference-cli --image $(IMAGE) --mask "$(MASK)"

health:
	curl -fsS http://localhost:$(PORT)/health | $(PYTHON) -m json.tool

clean:
	rm -rf $(WEB_DIR)/.next $(WEB_DIR)/tsconfig.tsbuildinfo
	rm -rf $(VENV)
	find $(INFERENCE_DIR) -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
