# Command façade. CMake stays the build system: no source lists, no compiler
# flags and no platform branches belong in here.
#
# Windows hosts without GNU make run the presets directly, which is what CI
# does too:  cmake --preset native-ci && cmake --build --preset native-ci
#            && ctest --preset native-ci
.PHONY: help configure build test test-native test-native-asan test-qt \
        test-python test-parity test-hygiene golden-freeze package clean \
        data-push data-push-dry-run data-pull data-pull-dry-run

PRESET ?= native-ci
BUCKET ?= hf://buckets/ubraket513/AlphaDiamond
DATA   ?= ./TrainAlphaDiamond

help:
	@echo "make test-native   native C++ tests, no Python required"
	@echo "make test-python   Python training/research suite"
	@echo "make test-parity   pybind boundary tests (needs pybind build)"
	@echo "make package       release package from the native-release preset"
	@echo "make data-push / data-pull (add -dry-run to preview)"
	@echo "make golden-freeze   re-record tests/golden provenance (contract change)"

configure:
	cmake --preset $(PRESET)

build: configure
	cmake --build --preset $(PRESET) --parallel

test: test-native

test-native: build
	ctest --preset $(PRESET) --output-on-failure

test-native-asan:
	$(MAKE) test-native PRESET=native-asan

test-qt:
	$(MAKE) test-native PRESET=native-qt

test-python:
	python -m pytest -m "not gui" --durations=10

test-parity:
	python -m pytest tests/native -v --durations=10

test-hygiene:
	python -m pytest tests/test_repo_hygiene.py -v

# The golden corpus is the normative game contract (docs/architecture/decisions.md).
# There is no regenerate target any more: the Python oracle that produced the
# corpus was deleted with the rest of the Python engine, and a deliberate
# contract change starts by restoring it from Git history, on purpose, and
# raising game_contract_version.
golden-freeze:
	python tools/freeze_golden.py

package:
	cmake --build --preset native-release --target package

clean:
	cmake -E rm -rf build

data-push-dry-run:
	hf sync $(DATA) $(BUCKET) --dry-run

data-push:
	hf sync $(DATA) $(BUCKET)

data-pull-dry-run:
	hf sync $(BUCKET) $(DATA) --dry-run

data-pull:
	hf sync $(BUCKET) $(DATA)
