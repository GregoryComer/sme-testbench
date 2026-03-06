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

void BM_gemm_qd8p_qc4wp_f32_4vlxvl(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  auto pack = sme::gemm_qd8_qc4w_4vlxvl_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  std::vector<float> C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s4(K, N, pack.rhs));
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s4(B.data(), B.size(), 123);

  std::vector<float> w_scales(N);
  fill_random_f32(w_scales.data(), N, 77);

  std::vector<float> w_ksums(N);
  sme::compute_ksums_s8(B.data(), K, N, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::QuantParams qp{0, 0.05f, w_scales.data(), w_ksums.data()};
  sme::pack_s8(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s4(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    sme::gemm_qd8p_qc4wp_f32_4vlxvl(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_gemm_qd8p_qc4wp_f32_4vlxvl)
    ->Args({128, 128, 128})
    ->Args({1024, 1024, 1024})
    ->Args({4096, 4096, 4096})
    ->Args({128, 128, 16384})
    ->Args({1024, 4096, 4096})
    ->Args({1024, 4096*4, 4096})
    ->Args({4096, 4096, 128000})
    ->Unit(benchmark::kMillisecond);

}  // namespace
