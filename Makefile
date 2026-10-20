.PHONY: test lint eval run

test:
	python -m pytest -q

lint:
	ruff check .

eval:
	python -m experiments.run_all

run:
	python -m backend.app
