SHELL=/bin/bash
LINT_PATHS=multi_type_feedback/ train_baselines/ setup.py

#pytest:
#	./scripts/run_tests.sh

mypy:
	mypy ${LINT_PATHS} --exclude 'setup.py'

missing-annotations:
	mypy --disallow-untyped-calls --disallow-untyped-defs --ignore-missing-imports multi_type_feedback

# missing docstrings
# pylint -d R,C,W,E -e C0116 rlhfblender -j 4

type: mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --show-files
	# exit-zero treats all errors as warnings.
	ruff check ${LINT_PATHS} --exit-zero

format:
	# Sort imports
	ruff check --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	ruff check --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format type lint

#doc:
#	cd docs && make html

spelling:
	cd docs && make spelling

clean:
	cd docs && make clean

.PHONY: clean spelling lint format check-codestyle commit-checks
