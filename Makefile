# file: Makefile

.PHONY: install api ui eval test lint format clean

install:
	pip install -r requirements.txt

api:
	uvicorn agentmesh.api.server:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run agentmesh/ui/app.py --server.port 8501

eval:
	python -m agentmesh.eval.cli run

test:
	pytest tests/ -v

lint:
	ruff check agentmesh/ tests/
	ruff format --check agentmesh/ tests/

format:
	ruff format agentmesh/ tests/

clean:
	rm -rf data/*.db data/*.bin data/sandbox/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete