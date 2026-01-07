.PHONY: all format lint test tests integration_tests docker_tests help extended_tests type_check type_check_package type_check_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE=tests/integration_tests/

test tests:
	uv run --group test pytest $(TEST_FILE)

integration_test integration_tests:
	uv run --group test --group test_integration pytest $(TEST_FILE)

test_watch:
	uv run --group test ptw --snapshot-update --now . -- -vv $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/alibabacloud_mysql --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_alibabacloud_mysql
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && uv run --all-groups mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --all-groups ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	uv run --group codespell codespell --toml pyproject.toml

spell_fix:
	uv run --group codespell codespell --toml pyproject.toml -w

check_imports: $(shell find langchain_alibabacloud_mysql -name '*.py')
	uv run --all-groups python ./scripts/check_imports.py $^

######################
# TYPE CHECKING
######################

# Define a variable for Python files to type check.
TYPE_CHECK_FILES=.
TYPE_CHECK_CACHE=.mypy_cache
type_check_package: TYPE_CHECK_FILES=langchain_alibabacloud_mysql
type_check_tests: TYPE_CHECK_FILES=tests
type_check_tests: TYPE_CHECK_CACHE=.mypy_cache_test

type_check type_check_package type_check_tests:
	[ "$(TYPE_CHECK_FILES)" = "" ] || mkdir -p $(TYPE_CHECK_CACHE) && uv run --all-groups mypy $(TYPE_CHECK_FILES) --cache-dir $(TYPE_CHECK_CACHE)

######################
# HELP
######################

help:
	@echo '----'
	@echo 'check_imports				- check imports'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'type_check                   - run type checker (mypy)'
	@echo 'type_check_package           - run type checker on package only'
	@echo 'type_check_tests             - run type checker on tests only'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'

