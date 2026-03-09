# SME Testbench

This repository contains various SME GEMM kernels, as well as test and benchmark harnesses.

## Building

Build with cmake.

```
mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

## Running Tests

Tests use gtest.

```
cd build
ctest
```

Note that there is no runtime feature detection. Attempting to run on hardware without relevant SME support will SIGILL.

## Running Benchmarks

Tests use Google benchmark. Each dtype permutation is built as a separate bench binary.

```
build/gemm_qd8_qb4w_bench # For 4-bit groupwise weights, 8-bit dynamic activations, f32 out
```