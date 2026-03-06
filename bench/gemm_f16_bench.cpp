#include "gemm.h"
#include "pack.h"

#include <algorithm>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>

namespace {

void fill_random(_Float16* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<_Float16>(dist(rng));
}

// --- 4vlxvl (4x1) single-threaded -------------------------------------------

void BM_gemm_f16p_f16p_f16_4vlxvl(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  auto pack = sme::gemm_f16_4vlxvl_packing_params();

  std::vector<_Float16> A(M * K), B(K * N), C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f16(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f16(K, N, pack.rhs));
  fill_random(A.data(), A.size(), 42);
  fill_random(B.data(), B.size(), 123);

  sme::GemmParams p{M, N, K};
  sme::pack_f16(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_f16(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    sme::gemm_f16p_f16p_f16_4vlxvl(p, lhs_packed.get(), rhs_packed.get(), C.data());
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_gemm_f16p_f16p_f16_4vlxvl)
    ->Args({128, 128, 128})
    ->Args({1024, 1024, 1024})
    ->Args({4096, 4096, 4096})
    ->Args({128, 128, 16384})
    ->Args({1024, 4096, 4096})
    ->Unit(benchmark::kMillisecond);

// --- 2vlx2vl (2x2) single-threaded -------------------------------------------

void BM_gemm_f16p_f16p_f16_2vlx2vl(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  auto pack = sme::gemm_f16_2vlx2vl_packing_params();

  std::vector<_Float16> A(M * K), B(K * N), C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f16(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f16(K, N, pack.rhs));
  fill_random(A.data(), A.size(), 42);
  fill_random(B.data(), B.size(), 123);

  sme::GemmParams p{M, N, K};
  sme::pack_f16(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_f16(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    sme::gemm_f16p_f16p_f16_2vlx2vl(p, lhs_packed.get(), rhs_packed.get(), C.data());
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_gemm_f16p_f16p_f16_2vlx2vl)
    ->Args({128, 128, 128})
    ->Args({1024, 1024, 1024})
    ->Args({4096, 4096, 4096})
    ->Args({128, 128, 16384})
    ->Args({1024, 4096, 4096})
    ->Unit(benchmark::kMillisecond);

// --- Multithreaded benchmark (M-split, 4vlxvl) ------------------------------

void gemm_f16p_f16p_f16_mt(const sme::GemmParams& p,
                             const void* lhs_packed, const void* rhs_packed,
                             _Float16* out, size_t num_threads) {
  auto pack = sme::gemm_f16_4vlxvl_packing_params();
  const size_t m_tile = pack.lhs.tile_rows;  // 4*vl_f32 = 64

  const size_t m_tiles = (p.M + m_tile - 1) / m_tile;
  const size_t tiles_per_thread = (m_tiles + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t t = 0; t < num_threads; t++) {
    const size_t m_start = t * tiles_per_thread * m_tile;
    if (m_start >= p.M) break;
    const size_t m_end = std::min((t + 1) * tiles_per_thread * m_tile, p.M);
    const size_t m_count = m_end - m_start;

    // LHS packed: tiles in M order, each tile is K * m_tile * sizeof(f16).
    const size_t lhs_offset_bytes =
        (m_start / m_tile) * p.K * m_tile * sizeof(_Float16);
    const auto* lhs_slice =
        static_cast<const char*>(lhs_packed) + lhs_offset_bytes;

    _Float16* out_slice = out + m_start * p.N;

    threads.emplace_back([=] {
      sme::GemmParams sp{m_count, p.N, p.K};
      sme::gemm_f16p_f16p_f16_4vlxvl(sp, lhs_slice, rhs_packed, out_slice);
    });
  }

  for (auto& th : threads) th.join();
}

void BM_gemm_f16p_f16p_f16_mt(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t num_threads = static_cast<size_t>(state.range(3));

  auto pack = sme::gemm_f16_4vlxvl_packing_params();

  std::vector<_Float16> A(M * K), B(K * N), C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f16(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f16(K, N, pack.rhs));
  fill_random(A.data(), A.size(), 42);
  fill_random(B.data(), B.size(), 123);

  sme::GemmParams p{M, N, K};
  sme::pack_f16(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_f16(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    gemm_f16p_f16p_f16_mt(p, lhs_packed.get(), rhs_packed.get(), C.data(),
                            num_threads);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["threads"] = static_cast<double>(num_threads);
}

BENCHMARK(BM_gemm_f16p_f16p_f16_mt)
    ->Args({4096, 4096, 4096, 1})
    ->Args({4096, 4096, 4096, 2})
    ->Args({4096, 4096, 4096, 4})
    ->Args({4096, 4096, 4096, 8})
    ->Args({128, 128, 16384, 1})
    ->Args({128, 128, 16384, 2})
    ->Args({128, 128, 16384, 4})
    ->Args({128, 128, 16384, 8})
    ->Args({1024, 4096, 4096, 1})
    ->Args({1024, 4096, 4096, 2})
    ->Args({1024, 4096, 4096, 4})
    ->Args({1024, 4096, 4096, 8})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->MinTime(0.3);

}  // namespace
