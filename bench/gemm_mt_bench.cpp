#include "gemm.h"
#include "pack.h"

#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>

namespace {

void fill_random(float* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

// --- Slice helpers for extracting submatrices from row-major matrices ---

// Extract columns [col_start, col_start+col_count) from a rows×cols matrix.
void slice_cols(const float* src, size_t rows, size_t cols,
                size_t col_start, size_t col_count, float* dst) {
  for (size_t r = 0; r < rows; r++) {
    std::memcpy(dst + r * col_count, src + r * cols + col_start,
                col_count * sizeof(float));
  }
}


// Multithreaded GEMM: split M across threads.
// Each thread enters streaming mode independently and processes its M-slice.
// The M-tile size for the 4x1 kernel is 4*svl (64 at SVL=512).
void gemm_f32p_f32p_f32_mt(const sme::GemmParams& p,
                            const void* lhs_packed, const void* rhs_packed,
                            float* out, size_t num_threads) {
  auto pack = sme::gemm_f32_packing_params();
  const size_t m_tile = pack.lhs.tile_rows;  // 4*svl = 64

  // Round up to tile boundary, then divide among threads.
  const size_t m_tiles = (p.M + m_tile - 1) / m_tile;
  const size_t tiles_per_thread = (m_tiles + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t t = 0; t < num_threads; t++) {
    const size_t m_start = t * tiles_per_thread * m_tile;
    if (m_start >= p.M) break;
    const size_t m_end_raw = (t + 1) * tiles_per_thread * m_tile;
    const size_t m_end = m_end_raw < p.M ? m_end_raw : p.M;
    const size_t m_count = m_end - m_start;

    // LHS packed layout: M-tiles in order, each tile is K * m_tile floats.
    const size_t lhs_offset_bytes =
        (m_start / m_tile) * p.K * m_tile * sizeof(float);
    const auto* lhs_slice =
        static_cast<const char*>(lhs_packed) + lhs_offset_bytes;

    // Output slice starts at row m_start.
    float* out_slice = out + m_start * p.N;

    threads.emplace_back([=] {
      sme::GemmParams slice_p{m_count, p.N, p.K};
      sme::gemm_f32p_f32p_f32(slice_p, lhs_slice, rhs_packed, out_slice);
    });
  }

  for (auto& t : threads) t.join();
}

void BM_gemm_f32p_f32p_f32_mt(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t num_threads = static_cast<size_t>(state.range(3));

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
    gemm_f32p_f32p_f32_mt(p, lhs_packed.get(), rhs_packed.get(), C.data(),
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

// Args: M, N, K, num_threads.
// Use large shapes where threading can help.
BENCHMARK(BM_gemm_f32p_f32p_f32_mt)
    // 4096x4096x4096 — large square baseline
    ->Args({4096, 4096, 4096, 1})
    ->Args({4096, 4096, 4096, 2})
    ->Args({4096, 4096, 4096, 4})
    ->Args({4096, 4096, 4096, 8})
    // 128x128x16384 — K-dominant
    ->Args({128, 128, 16384, 1})
    ->Args({128, 128, 16384, 2})
    ->Args({128, 128, 16384, 4})
    ->Args({128, 128, 16384, 8})
    // 1024x4096x4096 — large N
    ->Args({1024, 4096, 4096, 1})
    ->Args({1024, 4096, 4096, 2})
    ->Args({1024, 4096, 4096, 4})
    ->Args({1024, 4096, 4096, 8})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->MinTime(0.3);

// ---------------------------------------------------------------------------
// N-split: split N across threads.
// LHS packed once (shared). Each thread packs its own RHS column-slice and
// writes to an independent M × n_count output buffer.
// ---------------------------------------------------------------------------
void BM_gemm_f32p_f32p_f32_mt_nsplit(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));
  const size_t num_threads = static_cast<size_t>(state.range(3));

  auto pack = sme::gemm_f32_packing_params();
  const size_t n_tile = pack.rhs.tile_cols;

  std::vector<float> A(M * K), B(K * N);
  fill_random(A.data(), A.size(), 42);
  fill_random(B.data(), B.size(), 123);

  // Pack full LHS once (shared across threads).
  auto lhs_packed =
      std::make_unique<char[]>(sme::packed_size_bytes_f32(M, K, pack.lhs));
  sme::pack_f32(A.data(), M, K, pack.lhs, lhs_packed.get());

  // Per-thread data: packed RHS slice + output buffer.
  const size_t n_tiles = (N + n_tile - 1) / n_tile;
  const size_t tiles_per_thread = (n_tiles + num_threads - 1) / num_threads;

  struct ThreadData {
    std::unique_ptr<char[]> rhs_packed;
    std::vector<float> out;
    size_t n_count;
  };
  std::vector<ThreadData> td;

  for (size_t t = 0; t < num_threads; t++) {
    const size_t n_start = t * tiles_per_thread * n_tile;
    if (n_start >= N) break;
    const size_t n_end = std::min((t + 1) * tiles_per_thread * n_tile, N);
    const size_t n_count = n_end - n_start;

    std::vector<float> B_slice(K * n_count);
    slice_cols(B.data(), K, N, n_start, n_count, B_slice.data());

    ThreadData d;
    d.rhs_packed = std::make_unique<char[]>(
        sme::packed_size_bytes_f32(K, n_count, pack.rhs));
    sme::pack_f32(B_slice.data(), K, n_count, pack.rhs, d.rhs_packed.get());
    d.out.resize(M * n_count);
    d.n_count = n_count;
    td.push_back(std::move(d));
  }

  for (auto _ : state) {
    std::vector<std::thread> threads;
    threads.reserve(td.size());
    for (size_t t = 0; t < td.size(); t++) {
      threads.emplace_back([&, t] {
        sme::GemmParams sp{M, td[t].n_count, K};
        sme::gemm_f32p_f32p_f32(sp, lhs_packed.get(), td[t].rhs_packed.get(),
                                 td[t].out.data());
      });
    }
    for (auto& th : threads) th.join();
    for (auto& d : td) benchmark::DoNotOptimize(d.out.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * double(M) * double(N) * double(K);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["threads"] = static_cast<double>(num_threads);
}

BENCHMARK(BM_gemm_f32p_f32p_f32_mt_nsplit)
    // 4096x4096x4096 — large square baseline
    ->Args({4096, 4096, 4096, 1})
    ->Args({4096, 4096, 4096, 2})
    ->Args({4096, 4096, 4096, 4})
    ->Args({4096, 4096, 4096, 8})
    // 128x128x16384 — K-dominant
    ->Args({128, 128, 16384, 1})
    ->Args({128, 128, 16384, 2})
    ->Args({128, 128, 16384, 4})
    ->Args({128, 128, 16384, 8})
    // 1024x4096x4096 — large N
    ->Args({1024, 4096, 4096, 1})
    ->Args({1024, 4096, 4096, 2})
    ->Args({1024, 4096, 4096, 4})
    ->Args({1024, 4096, 4096, 8})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->MinTime(0.3);



}  // namespace
