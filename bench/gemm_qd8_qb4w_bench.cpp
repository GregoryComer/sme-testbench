#include "gemm.h"
#include "pack.h"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

namespace {

void fill_random_s8(int8_t* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-128, 127);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<int8_t>(dist(rng));
}

void fill_random_s4(int8_t* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-8, 7);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<int8_t>(dist(rng));
}

void fill_random_f32(float* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.01f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

// --- 4vlxvl (4x1) single-threaded -------------------------------------------

void BM_gemm_qd8p_qb4wp_f32_4vlxvl(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t group_size = static_cast<size_t>(state.range(3));

  size_t num_groups = (K + group_size - 1) / group_size;
  auto pack = sme::gemm_qd8_qb4w_4vlxvl_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  std::vector<float> C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s4(K, N, pack.rhs));
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s4(B.data(), B.size(), 123);

  std::vector<float> w_scales(num_groups * N);
  fill_random_f32(w_scales.data(), w_scales.size(), 77);

  std::vector<float> w_ksums(N);
  sme::compute_group_ksums_s8(B.data(), K, N, group_size, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::BlockQuantParams qp{0, 0.05f, group_size, w_scales.data(), w_ksums.data()};
  sme::pack_s8(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s4(B.data(), K, N, pack.rhs, rhs_packed.get());

  size_t svl_w = pack.rhs.tile_cols;
  std::vector<float> scratch_buf(4 * svl_w * svl_w);

  for (auto _ : state) {
    sme::gemm_qd8p_qb4wp_f32_4vlxvl(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp, scratch_buf.data());
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_gemm_qd8p_qb4wp_f32_4vlxvl)
    ->Args({128, 128, 128, 32})
    ->Args({128, 128, 128, 64})
    ->Args({128, 128, 128, 128})
    ->Args({128, 128, 128, 256})
    ->Args({1024, 1024, 1024, 32})
    ->Args({1024, 1024, 1024, 64})
    ->Args({1024, 1024, 1024, 128})
    ->Args({1024, 1024, 1024, 256})
    ->Args({4096, 4096, 4096, 32})
    ->Args({4096, 4096, 4096, 64})
    ->Args({4096, 4096, 4096, 128})
    ->Args({4096, 4096, 4096, 256})
    ->Args({128, 128, 16384, 32})
    ->Args({128, 128, 16384, 64})
    ->Args({128, 128, 16384, 128})
    ->Args({128, 128, 16384, 256})
    ->Args({1024, 4096, 4096, 32})
    ->Args({1024, 4096, 4096, 64})
    ->Args({1024, 4096, 4096, 128})
    ->Args({1024, 4096, 4096, 256})
    ->Args({1024, 4096*4, 4096, 32})
    ->Args({1024, 4096*4, 4096, 64})
    ->Args({1024, 4096*4, 4096, 128})
    ->Args({1024, 4096*4, 4096, 256})
    ->Args({4096, 4096, 128000, 32})
    ->Args({4096, 4096, 128000, 64})
    ->Args({4096, 4096, 128000, 128})
    ->Args({4096, 4096, 128000, 256})
    ->Unit(benchmark::kMillisecond);

// --- 2vlx2vl (2x2) single-threaded -------------------------------------------

void BM_gemm_qd8p_qb4wp_f32_2vlx2vl(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t group_size = static_cast<size_t>(state.range(3));

  size_t num_groups = (K + group_size - 1) / group_size;
  auto pack = sme::gemm_qd8_qb4w_2vlx2vl_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  std::vector<float> C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s4(K, N, pack.rhs));
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s4(B.data(), B.size(), 123);

  std::vector<float> w_scales(num_groups * N);
  fill_random_f32(w_scales.data(), w_scales.size(), 77);

  std::vector<float> w_ksums(N);
  sme::compute_group_ksums_s8(B.data(), K, N, group_size, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::BlockQuantParams qp{0, 0.05f, group_size, w_scales.data(), w_ksums.data()};
  sme::pack_s8(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s4(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    sme::gemm_qd8p_qb4wp_f32_2vlx2vl(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_gemm_qd8p_qb4wp_f32_2vlx2vl)
    ->Args({128, 128, 128, 32})
    ->Args({128, 128, 128, 64})
    ->Args({128, 128, 128, 128})
    ->Args({128, 128, 128, 256})
    ->Args({1024, 1024, 1024, 32})
    ->Args({1024, 1024, 1024, 64})
    ->Args({1024, 1024, 1024, 128})
    ->Args({1024, 1024, 1024, 256})
    ->Args({4096, 4096, 4096, 32})
    ->Args({4096, 4096, 4096, 64})
    ->Args({4096, 4096, 4096, 128})
    ->Args({4096, 4096, 4096, 256})
    ->Args({128, 128, 16384, 32})
    ->Args({128, 128, 16384, 64})
    ->Args({128, 128, 16384, 128})
    ->Args({128, 128, 16384, 256})
    ->Args({1024, 4096, 4096, 32})
    ->Args({1024, 4096, 4096, 64})
    ->Args({1024, 4096, 4096, 128})
    ->Args({1024, 4096, 4096, 256})
    ->Args({1024, 4096*4, 4096, 32})
    ->Args({1024, 4096*4, 4096, 64})
    ->Args({1024, 4096*4, 4096, 128})
    ->Args({1024, 4096*4, 4096, 256})
    ->Args({4096, 4096, 128000, 32})
    ->Args({4096, 4096, 128000, 64})
    ->Args({4096, 4096, 128000, 128})
    ->Args({4096, 4096, 128000, 256})
    ->Unit(benchmark::kMillisecond);

// --- 2vlxvl (2x1 SMOPA + ZA float accum) single-threaded --------------------

void BM_gemm_qd8p_qb4wp_f32_2vlxvl(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t group_size = static_cast<size_t>(state.range(3));

  size_t num_groups = (K + group_size - 1) / group_size;
  auto pack = sme::gemm_qd8_qb4w_2vlxvl_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  std::vector<float> C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s4(K, N, pack.rhs));
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s4(B.data(), B.size(), 123);

  std::vector<float> w_scales(num_groups * N);
  fill_random_f32(w_scales.data(), w_scales.size(), 77);

  std::vector<float> w_ksums(N);
  sme::compute_group_ksums_s8(B.data(), K, N, group_size, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::BlockQuantParams qp{0, 0.05f, group_size, w_scales.data(), w_ksums.data()};
  sme::pack_s8(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s4(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    sme::gemm_qd8p_qb4wp_f32_2vlxvl(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_gemm_qd8p_qb4wp_f32_2vlxvl)
    ->Args({128, 128, 128, 32})
    ->Args({128, 128, 128, 64})
    ->Args({128, 128, 128, 128})
    ->Args({128, 128, 128, 256})
    ->Args({1024, 1024, 1024, 32})
    ->Args({1024, 1024, 1024, 64})
    ->Args({1024, 1024, 1024, 128})
    ->Args({1024, 1024, 1024, 256})
    ->Args({4096, 4096, 4096, 32})
    ->Args({4096, 4096, 4096, 64})
    ->Args({4096, 4096, 4096, 128})
    ->Args({4096, 4096, 4096, 256})
    ->Args({128, 128, 16384, 32})
    ->Args({128, 128, 16384, 64})
    ->Args({128, 128, 16384, 128})
    ->Args({128, 128, 16384, 256})
    ->Args({1024, 4096, 4096, 32})
    ->Args({1024, 4096, 4096, 64})
    ->Args({1024, 4096, 4096, 128})
    ->Args({1024, 4096, 4096, 256})
    ->Args({1024, 4096*4, 4096, 32})
    ->Args({1024, 4096*4, 4096, 64})
    ->Args({1024, 4096*4, 4096, 128})
    ->Args({1024, 4096*4, 4096, 256})
    ->Args({4096, 4096, 128000, 32})
    ->Args({4096, 4096, 128000, 64})
    ->Args({4096, 4096, 128000, 128})
    ->Args({4096, 4096, 128000, 256})
    ->Unit(benchmark::kMillisecond);

// --- 2vlx2vl f16mopa (f16 widening FMOPA) single-threaded --------------------

void BM_gemm_qd8p_qb4wp_f32_2vlx2vl_f16mopa(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t group_size = static_cast<size_t>(state.range(3));

  size_t num_groups = (K + group_size - 1) / group_size;
  auto pack = sme::gemm_qd8_qb4w_2vlx2vl_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  std::vector<float> C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s4(K, N, pack.rhs));
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s4(B.data(), B.size(), 123);

  std::vector<float> w_scales(num_groups * N);
  fill_random_f32(w_scales.data(), w_scales.size(), 77);

  std::vector<float> w_ksums(N);
  sme::compute_group_ksums_s8(B.data(), K, N, group_size, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::BlockQuantParams qp{0, 0.05f, group_size, w_scales.data(), w_ksums.data()};
  sme::pack_s8_deinterleaved(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s4_deinterleaved(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    sme::gemm_qd8p_qb4wp_f32_2vlx2vl_f16mopa(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_gemm_qd8p_qb4wp_f32_2vlx2vl_f16mopa)
    ->Args({128, 128, 128, 32})
    ->Args({128, 128, 128, 64})
    ->Args({128, 128, 128, 128})
    ->Args({128, 128, 128, 256})
    ->Args({1024, 1024, 1024, 32})
    ->Args({1024, 1024, 1024, 64})
    ->Args({1024, 1024, 1024, 128})
    ->Args({1024, 1024, 1024, 256})
    ->Args({4096, 4096, 4096, 32})
    ->Args({4096, 4096, 4096, 64})
    ->Args({4096, 4096, 4096, 128})
    ->Args({4096, 4096, 4096, 256})
    ->Args({128, 128, 16384, 32})
    ->Args({128, 128, 16384, 64})
    ->Args({128, 128, 16384, 128})
    ->Args({128, 128, 16384, 256})
    ->Args({1024, 4096, 4096, 32})
    ->Args({1024, 4096, 4096, 64})
    ->Args({1024, 4096, 4096, 128})
    ->Args({1024, 4096, 4096, 256})
    ->Args({1024, 4096*4, 4096, 32})
    ->Args({1024, 4096*4, 4096, 64})
    ->Args({1024, 4096*4, 4096, 128})
    ->Args({1024, 4096*4, 4096, 256})
    ->Args({4096, 4096, 128000, 32})
    ->Args({4096, 4096, 128000, 64})
    ->Args({4096, 4096, 128000, 128})
    ->Args({4096, 4096, 128000, 256})
    ->Unit(benchmark::kMillisecond);

// --- 2vlxvl 2-level block scales (int8 inner + f32 outer) --------------------

void fill_random_s8_range(int8_t* buf, size_t n, unsigned seed, int lo, int hi) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(lo, hi);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<int8_t>(dist(rng));
}

void BM_gemm_qd8p_qb4w2lp_f32_2vlxvl(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t inner_group_size = static_cast<size_t>(state.range(3));
  const size_t outer_group_size = static_cast<size_t>(state.range(4));

  size_t num_inner = K / inner_group_size;
  size_t num_outer = K / outer_group_size;
  auto pack = sme::gemm_qd8_qb4w2l_2vlxvl_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  std::vector<float> C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s4(K, N, pack.rhs));
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s4(B.data(), B.size(), 123);

  std::vector<int8_t> inner_scales(num_inner * N);
  fill_random_s8_range(inner_scales.data(), inner_scales.size(), 200, 1, 15);

  std::vector<float> outer_scales(num_outer * N);
  fill_random_f32(outer_scales.data(), outer_scales.size(), 300);

  std::vector<float> w_ksums(N);
  sme::compute_group_ksums_2level(B.data(), K, N, inner_group_size, outer_group_size,
      inner_scales.data(), outer_scales.data(), w_ksums.data());

  size_t svl_w = pack.rhs.tile_cols;
  std::vector<int8_t> packed_inner(
      sme::packed_group_scales_s8_len(num_inner, N, svl_w));
  sme::pack_group_scales_s8(inner_scales.data(), num_inner, N, svl_w,
                             packed_inner.data());

  std::vector<float> packed_outer(
      sme::packed_group_scales_len(num_outer, N, svl_w));
  sme::pack_group_scales(outer_scales.data(), num_outer, N, svl_w,
                          packed_outer.data());

  sme::GemmParams p{M, N, K};
  sme::BlockQuantParams2L qp{0, 0.05f, inner_group_size, outer_group_size,
                              packed_inner.data(), packed_outer.data(),
                              w_ksums.data()};
  sme::pack_s8(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s4(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    sme::gemm_qd8p_qb4w2lp_f32_2vlxvl(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["inner_gs"] = static_cast<double>(inner_group_size);
  state.counters["outer_gs"] = static_cast<double>(outer_group_size);
}

// Args: M, N, K, inner_group_size, outer_group_size
// Sweep outer group sizes for inner=32
#define OGS_SWEEP_32(M, N, K) \
    ->Args({M, N, K, 32, 128})   \
    ->Args({M, N, K, 32, 256})   \
    ->Args({M, N, K, 32, 512})   \
    ->Args({M, N, K, 32, 1024})  \
    ->Args({M, N, K, 32, 2048})  \
    ->Args({M, N, K, 32, 4096})

// Sweep outer group sizes for inner=128
#define OGS_SWEEP_128(M, N, K) \
    ->Args({M, N, K, 128, 128})   \
    ->Args({M, N, K, 128, 256})   \
    ->Args({M, N, K, 128, 512})   \
    ->Args({M, N, K, 128, 1024})  \
    ->Args({M, N, K, 128, 2048})  \
    ->Args({M, N, K, 128, 4096})

BENCHMARK(BM_gemm_qd8p_qb4w2lp_f32_2vlxvl)
    OGS_SWEEP_32(1024, 1024, 4096)
    OGS_SWEEP_32(4096, 4096, 4096)
    OGS_SWEEP_32(1024, 4096, 4096)
    OGS_SWEEP_32(128, 4096, 4096)
    OGS_SWEEP_128(1024, 1024, 4096)
    OGS_SWEEP_128(4096, 4096, 4096)
    OGS_SWEEP_128(1024, 4096, 4096)
    OGS_SWEEP_128(128, 4096, 4096)
    ->Unit(benchmark::kMillisecond);

}  // namespace
