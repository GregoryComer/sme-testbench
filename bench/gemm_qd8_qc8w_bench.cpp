#include "gemm.h"
#include "pack.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>

namespace {

void fill_random_s8(int8_t* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-128, 127);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<int8_t>(dist(rng));
}

void fill_random_f32(float* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.01f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

// --- 4vlxvl (4x1) single-threaded -------------------------------------------

void BM_gemm_qd8p_qc8wp_f32_4vlxvl(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  auto pack = sme::gemm_qd8_qc8w_4vlxvl_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  std::vector<float> C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(K, N, pack.rhs));
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s8(B.data(), B.size(), 123);

  std::vector<float> w_scales(N);
  fill_random_f32(w_scales.data(), N, 77);

  std::vector<float> w_ksums(N);
  sme::compute_ksums_s8(B.data(), K, N, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::QuantParams qp{0, 0.05f, w_scales.data(), w_ksums.data()};
  sme::pack_s8(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s8(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    sme::gemm_qd8p_qc8wp_f32_4vlxvl(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_gemm_qd8p_qc8wp_f32_4vlxvl)
    ->Args({128, 128, 128})
    ->Args({1024, 1024, 1024})
    ->Args({4096, 4096, 4096})
    ->Args({128, 128, 16384})
    ->Args({1024, 4096, 4096})
    ->Args({1024, 4096*4, 4096})
    ->Args({4096, 4096, 128000})
    ->Unit(benchmark::kMillisecond);

// --- 2vlx2vl (2x2) single-threaded -------------------------------------------

void BM_gemm_qd8p_qc8wp_f32_2vlx2vl(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  auto pack = sme::gemm_qd8_qc8w_2vlx2vl_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  std::vector<float> C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(K, N, pack.rhs));
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s8(B.data(), B.size(), 123);

  std::vector<float> w_scales(N);
  fill_random_f32(w_scales.data(), N, 77);

  std::vector<float> w_ksums(N);
  sme::compute_ksums_s8(B.data(), K, N, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::QuantParams qp{0, 0.05f, w_scales.data(), w_ksums.data()};
  sme::pack_s8(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s8(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    sme::gemm_qd8p_qc8wp_f32_2vlx2vl(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                 static_cast<double>(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_gemm_qd8p_qc8wp_f32_2vlx2vl)
    ->Args({128, 128, 128})
    ->Args({1024, 1024, 1024})
    ->Args({4096, 4096, 4096})
    ->Args({128, 128, 16384})
    ->Args({1024, 4096, 4096})
    ->Args({1024, 4096*4, 4096})
    ->Unit(benchmark::kMillisecond);

// --- Multithreaded benchmark (M-split, 4vlxvl) ------------------------------

void gemm_qd8p_qc8wp_f32_mt(const sme::GemmParams& p, const sme::QuantParams& qp,
                           const void* lhs_packed, const void* rhs_packed,
                           float* out, size_t num_threads) {
  auto pack = sme::gemm_qd8_qc8w_4vlxvl_packing_params();
  const size_t m_tile = pack.lhs.tile_rows;

  const size_t m_tiles = (p.M + m_tile - 1) / m_tile;
  const size_t tiles_per_thread = (m_tiles + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t t = 0; t < num_threads; t++) {
    const size_t m_start = t * tiles_per_thread * m_tile;
    if (m_start >= p.M) break;
    const size_t m_end = std::min((t + 1) * tiles_per_thread * m_tile, p.M);
    const size_t m_count = m_end - m_start;

    const size_t lhs_offset_bytes =
        (m_start / m_tile) * p.K * m_tile * sizeof(int8_t);
    const auto* lhs_slice =
        static_cast<const char*>(lhs_packed) + lhs_offset_bytes;

    float* out_slice = out + m_start * p.N;

    threads.emplace_back([=] {
      sme::GemmParams sp{m_count, p.N, p.K};
      sme::gemm_qd8p_qc8wp_f32_4vlxvl(sp, lhs_slice, rhs_packed, out_slice, qp);
    });
  }

  for (auto& th : threads) th.join();
}

void BM_gemm_qd8p_qc8wp_f32_mt(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t num_threads = static_cast<size_t>(state.range(3));

  auto pack = sme::gemm_qd8_qc8w_4vlxvl_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  std::vector<float> C(M * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(K, N, pack.rhs));
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s8(B.data(), B.size(), 123);

  std::vector<float> w_scales(N);
  fill_random_f32(w_scales.data(), N, 77);

  std::vector<float> w_ksums(N);
  sme::compute_ksums_s8(B.data(), K, N, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::QuantParams qp{0, 0.05f, w_scales.data(), w_ksums.data()};
  sme::pack_s8(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s8(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    gemm_qd8p_qc8wp_f32_mt(p, qp, lhs_packed.get(), rhs_packed.get(), C.data(),
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

BENCHMARK(BM_gemm_qd8p_qc8wp_f32_mt)
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
