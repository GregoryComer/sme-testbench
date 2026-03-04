cmake_minimum_required(VERSION 3.20)
project(sme-testbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------- library ----------
add_library(sme_kernels
  src/pack.cpp
  src/gemm.cpp
  src/gemm_f32_sme.cpp
  src/reference.cpp
)
target_include_directories(sme_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# SME flags only on the kernel file. Non-streaming code must not see +sve
# since Apple Silicon only supports SVE in streaming mode.
set_source_files_properties(src/gemm_f32_sme.cpp PROPERTIES
  COMPILE_OPTIONS "-march=armv8-a+sme"
)

# ---------- dependencies ----------
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest googlebenchmark)

# ---------- tests ----------
enable_testing()

add_executable(gemm_test test/gemm_test.cpp)
target_link_libraries(gemm_test PRIVATE sme_kernels GTest::gtest_main)

add_executable(pack_test test/pack_test.cpp)
target_link_libraries(pack_test PRIVATE sme_kernels GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(gemm_test)
gtest_discover_tests(pack_test)

# ---------- benchmarks ----------
add_executable(gemm_bench bench/gemm_bench.cpp)
target_link_libraries(gemm_bench PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(gemm_mt_bench bench/gemm_mt_bench.cpp)
target_link_libraries(gemm_mt_bench PRIVATE sme_kernels benchmark::benchmark_main)
