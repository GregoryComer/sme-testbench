cmake_minimum_required(VERSION 3.20)
project(sme-testbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------- library ----------
add_library(sme_kernels
  src/pack.cpp
  src/gemm.cpp
  src/gemm_f32_4vlxvl_sme.cpp
  src/gemm_f32_2vlx2vl_sme.cpp
  src/gemm_f16_4vlxvl_sme.cpp
  src/gemm_f16_2vlx2vl_sme.cpp
  src/gemm_bf16_4vlxvl_sme.cpp
  src/gemm_bf16_2vlx2vl_sme.cpp
  src/gemm_qd8_qc8w_4vlxvl_sme.cpp
  src/gemm_qd8_qc8w_2vlx2vl_sme.cpp
  src/gemm_qd8_qc4w_4vlxvl_sme.cpp
  src/gemm_qd8_qb4w_4vlxvl_sme.cpp
  src/gemm_qd8_qb4w_2vlx2vl_sme.cpp
  src/gemm_qd8_qb4w_2vlxvl_sme.cpp
  src/gemm_qd8_qb4w_2vlx2vl_f16mopa.cpp
  src/gemm_qd8_qb4w2l_2vlxvl_sme.cpp
  src/reference.cpp
)
target_include_directories(sme_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# SME flags only on kernel files. Non-streaming code must not see +sve
# since Apple Silicon only supports SVE in streaming mode.
set_source_files_properties(
  src/gemm_f32_4vlxvl_sme.cpp
  src/gemm_f32_2vlx2vl_sme.cpp
  src/gemm_f16_4vlxvl_sme.cpp
  src/gemm_f16_2vlx2vl_sme.cpp
  src/gemm_bf16_4vlxvl_sme.cpp
  src/gemm_bf16_2vlx2vl_sme.cpp
  src/gemm_qd8_qc8w_4vlxvl_sme.cpp
  src/gemm_qd8_qc8w_2vlx2vl_sme.cpp
  src/gemm_qd8_qc4w_4vlxvl_sme.cpp
  src/gemm_qd8_qb4w_4vlxvl_sme.cpp
  src/gemm_qd8_qb4w_2vlx2vl_sme.cpp
  src/gemm_qd8_qb4w_2vlxvl_sme.cpp
  src/gemm_qd8_qb4w2l_2vlxvl_sme.cpp
  PROPERTIES COMPILE_OPTIONS "-march=armv8-a+sme"
)

# f16 widening FMOPA kernel (svmopa_za32_f16_m is SME1).
set_source_files_properties(
  src/gemm_qd8_qb4w_2vlx2vl_f16mopa.cpp
  PROPERTIES COMPILE_OPTIONS "-march=armv8-a+sme"
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

add_executable(gemm_f16_test test/gemm_f16_test.cpp)
target_link_libraries(gemm_f16_test PRIVATE sme_kernels GTest::gtest_main)

add_executable(pack_test test/pack_test.cpp)
target_link_libraries(pack_test PRIVATE sme_kernels GTest::gtest_main)

add_executable(gemm_bf16_test test/gemm_bf16_test.cpp)
target_link_libraries(gemm_bf16_test PRIVATE sme_kernels GTest::gtest_main)

add_executable(gemm_qd8_qc8w_test test/gemm_qd8_qc8w_test.cpp)
target_link_libraries(gemm_qd8_qc8w_test PRIVATE sme_kernels GTest::gtest_main)

add_executable(gemm_qd8_qc4w_test test/gemm_qd8_qc4w_test.cpp)
target_link_libraries(gemm_qd8_qc4w_test PRIVATE sme_kernels GTest::gtest_main)

add_executable(gemm_qd8_qb4w_test test/gemm_qd8_qb4w_test.cpp)
target_link_libraries(gemm_qd8_qb4w_test PRIVATE sme_kernels GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(gemm_test)
gtest_discover_tests(gemm_f16_test)
gtest_discover_tests(gemm_bf16_test)
gtest_discover_tests(gemm_qd8_qc8w_test)
gtest_discover_tests(gemm_qd8_qc4w_test)
gtest_discover_tests(gemm_qd8_qb4w_test)
gtest_discover_tests(pack_test)

# ---------- benchmarks ----------
add_executable(gemm_bench bench/gemm_bench.cpp)
target_link_libraries(gemm_bench PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(gemm_f16_bench bench/gemm_f16_bench.cpp)
target_link_libraries(gemm_f16_bench PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(gemm_bf16_bench bench/gemm_bf16_bench.cpp)
target_link_libraries(gemm_bf16_bench PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(gemm_qd8_qc8w_bench bench/gemm_qd8_qc8w_bench.cpp)
target_link_libraries(gemm_qd8_qc8w_bench PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(gemm_qd8_qc4w_bench bench/gemm_qd8_qc4w_bench.cpp)
target_link_libraries(gemm_qd8_qc4w_bench PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(gemm_qd8_qb4w_bench bench/gemm_qd8_qb4w_bench.cpp)
target_link_libraries(gemm_qd8_qb4w_bench PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(gemm_mt_bench bench/gemm_mt_bench.cpp)
target_link_libraries(gemm_mt_bench PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(qd8_qc8w_pf_sweep bench/qd8_qc8w_pf_sweep.cpp)
set_source_files_properties(bench/qd8_qc8w_pf_sweep.cpp
  PROPERTIES COMPILE_OPTIONS "-march=armv8-a+sme"
)
target_link_libraries(qd8_qc8w_pf_sweep PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(tile_2x2_bench bench/tile_2x2_bench.cpp)
set_source_files_properties(bench/tile_2x2_bench.cpp
  PROPERTIES COMPILE_OPTIONS "-march=armv8-a+sme"
)
target_link_libraries(tile_2x2_bench PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(qd8_qc8w_2x2_bench bench/qd8_qc8w_2x2_bench.cpp)
set_source_files_properties(bench/qd8_qc8w_2x2_bench.cpp
  PROPERTIES COMPILE_OPTIONS "-march=armv8-a+sme"
)
target_link_libraries(qd8_qc8w_2x2_bench PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(f32_pf_sweep bench/f32_pf_sweep.cpp)
set_source_files_properties(bench/f32_pf_sweep.cpp
  PROPERTIES COMPILE_OPTIONS "-march=armv8-a+sme"
)
target_link_libraries(f32_pf_sweep PRIVATE sme_kernels benchmark::benchmark_main)

add_executable(mopa_peak_bench bench/mopa_peak_bench.cpp)
set_source_files_properties(bench/mopa_peak_bench.cpp
  PROPERTIES COMPILE_OPTIONS "-march=armv8-a+sme"
)
target_link_libraries(mopa_peak_bench PRIVATE sme_kernels)
