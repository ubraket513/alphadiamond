# Command façade. CMake stays the build system: no source lists, no compiler
# flags and no platform branches belong in here.
#
# Windows Git Bash and Unix hosts use the same launcher and presets.
.PHONY: help configure build test test-native test-training test-native-asan \
        test-qt package

PRESET ?= native-ci
NATIVE_TOOL := tools/native_training.sh

help:
	@echo "make test-native     native C++ contract tests"
	@echo "make test-training   native LibTorch training tests"
	@echo "make test-qt         Qt + LibTorch application tests"
	@echo "make package         native Windows application package"

configure:
	$(NATIVE_TOOL) cmake --preset $(PRESET)

build: configure
	$(NATIVE_TOOL) cmake --build --preset $(PRESET) --parallel

test: test-native

test-native: build
	$(NATIVE_TOOL) ctest --preset $(PRESET) --output-on-failure

test-training:
	$(MAKE) test-native PRESET=native-training

test-native-asan:
	$(MAKE) test-native PRESET=native-asan

test-qt:
	$(MAKE) test-native PRESET=native-qt

package:
	$(NATIVE_TOOL) cmake --preset native-package
	$(NATIVE_TOOL) cmake --build --preset native-package --target package --parallel
