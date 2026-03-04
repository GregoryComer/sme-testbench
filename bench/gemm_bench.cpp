#include "gemm.h"
#include "pack.h"

#include <memory>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

namespace {

void fill_random(float* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

void BM_gemm_f32p_f32p_f32(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  auto pack = sme::gemm_f32_packing_params();

  std::vector<float> A(M * K), B(K * N), C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f32(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f32(K, N, pack.rhs));
  fill_random(A.data(), A.size(), 42);
  fill_random(B.data(), B.size(), 123);

  sme::GemmParams p{M, N, K};
  sme::pack_f32(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_f32(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    sme::gemm_f32p_f32p_f32(p, lhs_packed.get(), rhs_packed.get(), C.data());
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  // 2*M*N*K FLOPs per GEMM (multiply + add).
  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

// M values: 1 (GEMV-like), 16, 128, 1024.
// N, K values: 1024, 4096.
BENCHMARK(BM_gemm_f32p_f32p_f32)
    ->Args({1, 1024, 1024})
    ->Args({1, 4096, 4096})
    ->Args({16, 1024, 1024})
    ->Args({16, 4096, 4096})
    ->Args({128, 1024, 1024})
    ->Args({128, 4096, 4096})
    ->Args({1024, 1024, 1024})
    ->Args({1024, 4096, 4096})
    ->Threads(1)
    ->Unit(benchmark::kMillisecond);

}  // namespace
