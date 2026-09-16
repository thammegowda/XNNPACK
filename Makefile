# Native builds of the library and registered CMake tests.
.DEFAULT_GOAL := release
.NOTPARALLEL:

TEST_JOBS ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)

export CC CXX
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1

build_type_debug := Debug
build_type_release := Release
build_tests_debug := ON
build_tests_release := OFF

.PHONY: debug release test

debug release:
	cmake -S . -B "build/$@" -G Ninja \
	  -DCMAKE_BUILD_TYPE=$(build_type_$@) \
	  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	  -DXNNPACK_LIBRARY_TYPE=static -DXNNPACK_BUILD_LIBRARY=ON \
	  -DXNNPACK_BUILD_TESTS=$(build_tests_$@) \
	  -DXNNPACK_BUILD_AMX_TESTS=$(build_tests_$@) \
	  -DXNNPACK_BUILD_ALL_MICROKERNELS=$(build_tests_$@) \
	  -DXNNPACK_BUILD_BENCHMARKS=OFF $(CMAKE_ARGS)
	cmake --build "build/$@" -j $(BUILD_ARGS)

test:
	ctest --test-dir "build/debug" \
	  --output-on-failure --no-tests=error --parallel $(TEST_JOBS) $(CTEST_ARGS)
