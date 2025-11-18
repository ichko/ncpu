# Detect OS and set PYTHON command accordingly
ifeq ($(OS),Windows_NT)
    PYTHON := $(shell where python3)
else
    PYTHON := $(shell which python3)
endif

SOURCES := src

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