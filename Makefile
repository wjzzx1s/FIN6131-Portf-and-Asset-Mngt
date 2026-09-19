# All Python commands use the environment inside this repository.
PY := $(CURDIR)/.venv/bin/python
A := assignment1
.NOTPARALLEL:
.PHONY: all setup data test notebook report verify clean-cache

all: test notebook report verify

setup:
	test -x "$(PY)" || uv venv .venv --python python3
	uv pip install --python $(PY) -r requirements.lock.txt

# Reconstruct from archived real provider responses (no network).
data:
	$(PY) $(A)/data_io.py

test:
	$(PY) -m pytest -q

# Regenerate before execution to keep the embedded tested core up to date.
notebook:
	$(PY) $(A)/build_notebook.py
	$(PY) $(A)/execute_notebook.py

report:
	cd $(A) && latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex

verify:
	$(PY) $(A)/verify_artifacts.py

# Never deletes the report, notebook, cached data, or figures.
clean-cache:
	cd $(A) && latexmk -c report.tex
