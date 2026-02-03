# Detect OS and set PYTHON command accordingly
PYTHON := uv run
SOURCES := src
TEST_DIR := tests

lint:
	${PYTHON} -m flake8 ${SOURCES}
	${PYTHON} -m black --check ${SOURCES}

format:
	${PYTHON} -m autoflake --remove-all-unused-imports --in-place --recursive ${SOURCES}
	${PYTHON} -m black ${SOURCES}

# we can activate it later if needed
# test:
# 	${PYTHON} -m unittest tests/test_*

generate_reqs:
	${PYTHON} -m pipreqs . --force

clean_checkpoints: 
	rm -rf notebooks/checkpoints/*

test:
	${PYTHON} -m unittest discover -s ${TEST_DIR} -p "test_*.py"
