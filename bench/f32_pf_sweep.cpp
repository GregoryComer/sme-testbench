// Prefetch distance sweep for f32 FMOPA GEMM kernel.
// PF_DIST is in K-steps. Per K-step byte distances:
//   LHS = 4*vl*sizeof(float) = 4*16*4 = 256B
//   RHS = vl*sizeof(float)   = 16*4   =  64B
// Sweeps PF_DIST = 0, 16, 32, 64, 128, 256, 512, 1024.

#include "gemm.h"
#include "pack.h"

#include <arm_sme.h>
#include <cstdint>
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

// ---------------------------------------------------------------------------
// Kernel template — inlined f32 4x1 micro-kernel with compile-time PF_DIST.
// ---------------------------------------------------------------------------

template <int PF_DIST>
__attribute__((noinline))
void f32_kernel(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out) __arm_streaming __arm_inout("za") {

  const auto* lhs_base = static_cast<const float*>(lhs_packed);
  const auto* rhs_base = static_cast<const float*>(rhs_packed);
  const uint64_t vl = svcntw();
  const svbool_t pg = svptrue_b32();

  const uint64_t m_body = (p.M / (vl * 4)) * (vl * 4);

  for (uint64_t m = 0; m < m_body; m += vl * 4) {
    const float* lhs_m = lhs_base + (m / (vl * 4)) * p.K * vl * 4;

    for (uint64_t n = 0; n < p.N; n += vl) {
      const float* rhs_n = rhs_base + (n / vl) * p.K * vl;

      svzero_za();

      const float* lhs_data = lhs_m;
      const float* rhs_data = rhs_n;

      for (uint64_t k = 0; k < p.K; k++) {
        if constexpr (PF_DIST > 0) {
          svprfb(pg, lhs_data + PF_DIST * vl * 4, SV_PLDL1KEEP);
          svprfb(pg, rhs_data + PF_DIST * vl, SV_PLDL1KEEP);
        }

        svfloat32_t lhs_col0 = svld1_f32(pg, lhs_data);
        svfloat32_t lhs_col1 = svld1_f32(pg, lhs_data + vl);
        svfloat32_t lhs_col2 = svld1_f32(pg, lhs_data + vl * 2);
        svfloat32_t lhs_col3 = svld1_f32(pg, lhs_data + vl * 3);
        svfloat32_t rhs_row0 = svld1_f32(pg, rhs_data);

        svmopa_za32_f32_m(0, pg, pg, lhs_col0, rhs_row0);
        svmopa_za32_f32_m(1, pg, pg, lhs_col1, rhs_row0);
        svmopa_za32_f32_m(2, pg, pg, lhs_col2, rhs_row0);
        svmopa_za32_f32_m(3, pg, pg, lhs_col3, rhs_row0);

        lhs_data += vl * 4;
        rhs_data += vl;
      }

      const svbool_t n_pg = svwhilelt_b32(n, (uint64_t)p.N);
      float* out0 = out + m * p.N + n;
      float* out1 = out + (m + vl) * p.N + n;
      float* out2 = out + (m + vl * 2) * p.N + n;
      float* out3 = out + (m + vl * 3) * p.N + n;

      for (uint32_t i = 0; i < vl; i++) {
        svst1_hor_za32(0, i, n_pg, out0 + i * p.N);
        svst1_hor_za32(1, i, n_pg, out1 + i * p.N);
        svst1_hor_za32(2, i, n_pg, out2 + i * p.N);
        svst1_hor_za32(3, i, n_pg, out3 + i * p.N);
      }
    }
  }

  // M epilogue: remaining rows (< 4*vl), 1×1 predicated.
  if (m_body < p.M) {
    const float* lhs_m = lhs_base + (m_body / (vl * 4)) * p.K * vl * 4;

    for (uint64_t s = 0; m_body + s * vl < p.M; s++) {
      const uint64_t m_row = m_body + s * vl;
      const svbool_t m_pred = svwhilelt_b32(m_row, (uint64_t)p.M);
      const uint64_t rem = p.M - m_row;
      const uint64_t rows = rem < vl ? rem : vl;

      for (uint64_t n = 0; n < p.N; n += vl) {
        const float* rhs_n = rhs_base + (n / vl) * p.K * vl;

        svzero_za();

        const float* lhs_data = lhs_m + s * vl;
        const float* rhs_data = rhs_n;

        for (uint64_t k = 0; k < p.K; k++) {
          svfloat32_t lhs_col = svld1_f32(m_pred, lhs_data);
          svfloat32_t rhs_row = svld1_f32(pg, rhs_data);
          svmopa_za32_f32_m(0, m_pred, pg, lhs_col, rhs_row);
          lhs_data += vl * 4;
          rhs_data += vl;
        }

        const svbool_t n_pg = svwhilelt_b32(n, (uint64_t)p.N);
        float* out_ptr = out + m_row * p.N + n;
        for (uint32_t i = 0; i < rows; i++) {
          svst1_hor_za32(0, i, n_pg, out_ptr + i * p.N);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Non-streaming wrapper.
// ---------------------------------------------------------------------------

template <int PF_DIST>
__attribute__((noinline)) __arm_locally_streaming __arm_new("za")
void f32_run(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out) {
  f32_kernel<PF_DIST>(p, lhs_packed, rhs_packed, out);
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

template <int PF_DIST>
void BM_f32_pf(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  auto pack = sme::gemm_f32_4vlxvl_packing_params();

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
    f32_run<PF_DIST>(p, lhs_packed.get(), rhs_packed.get(), C.data());
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * double(M) * double(N) * double(K);
  state.counters["GFLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  state.counters["pf"] = PF_DIST;
}

#define SIZES \
  ->Args({1024, 1024, 1024})  \
  ->Args({4096, 4096, 4096})  \
  ->Args({1024, 4096, 4096})  \
  ->Unit(benchmark::kMillisecond)

BENCHMARK(BM_f32_pf<0>)    SIZES;
BENCHMARK(BM_f32_pf<16>)   SIZES;
BENCHMARK(BM_f32_pf<32>)   SIZES;
BENCHMARK(BM_f32_pf<64>)   SIZES;
BENCHMARK(BM_f32_pf<128>)  SIZES;
BENCHMARK(BM_f32_pf<256>)  SIZES;
BENCHMARK(BM_f32_pf<512>)  SIZES;
BENCHMARK(BM_f32_pf<1024>) SIZES;

}  // namespace
