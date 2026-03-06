// Prefetch distance sweep for qd8_qc8w GEMM kernel.
// PF_DIST is in K-steps (matching the f32 kernel convention).
// Per K-step byte distances: LHS = 4*svcntb = 256B, RHS = svcntb = 64B.
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

// ---------------------------------------------------------------------------
// Kernel template — follows the f32 prefetch pattern.
// PF_DIST is in K-steps ahead (each K-step = 4 K-values for rank-4 SMOPA).
// ---------------------------------------------------------------------------

template <int PF_DIST>
__attribute__((noinline))
void qc8w_kernel(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out, const sme::QuantParams& qp) __arm_streaming __arm_inout("za") {

  size_t m = 0;
  auto lhs = static_cast<const int8_t*>(lhs_packed);
  auto p_K_padded = (p.K + 3) & ~size_t(0x3);
  svbool_t pg = svptrue_b8();

  // --- Main body: 4 M-subtiles × 1 N-tile ---
  for (; m + (svcntw() * 4) <= p.M; m += svcntw() * 4) {
    auto rhs_data = static_cast<const int8_t*>(rhs_packed);

    for (size_t n = 0; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;
      svzero_za();

      for (size_t k = 0; k < p.K; k += 4) {
#if PF_DIST > 0
        // Prefetch is only emitted when PF_DIST > 0 at compile time,
        // but we use 'if constexpr' for the template parameter.
#endif
        if constexpr (PF_DIST > 0) {
          svprfb(pg, lhs_data + PF_DIST * svcntb() * 4, SV_PLDL1KEEP);
          svprfb(pg, rhs_data + PF_DIST * svcntb(), SV_PLDL1KEEP);
        }

        auto l0 = svld1_s8(pg, lhs_data);
        auto l1 = svld1_s8(pg, lhs_data + svcntb());
        auto l2 = svld1_s8(pg, lhs_data + svcntb() * 2);
        auto l3 = svld1_s8(pg, lhs_data + svcntb() * 3);
        auto r0 = svld1_s8(pg, rhs_data);

        svmopa_za32_s8_m(0, pg, pg, l0, r0);
        svmopa_za32_s8_m(1, pg, pg, l1, r0);
        svmopa_za32_s8_m(2, pg, pg, l2, r0);
        svmopa_za32_s8_m(3, pg, pg, l3, r0);

        lhs_data += 4 * svcntb();
        rhs_data += svcntb();
      }

      // Dequant + store.
      auto a_scale = svdup_n_f32(qp.a_scale);
      auto w_scales0 = svld1_f32(pg, &qp.w_scales[n]);
      auto scales0 = svmul_f32_x(pg, w_scales0, a_scale);
      auto ksums0 = svld1_f32(pg, &qp.w_ksums[n]);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg, svmul_f32_x(pg, ksums0, zp_f32), a_scale);

      float* out_ptr = out + m * p.N + n;
      svbool_t n_pred = svwhilelt_b32(n, (uint64_t)p.N);

      for (uint32_t i = 0; i < svcntw(); i++) {
        svint32_t row0 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i);
        svint32_t row1 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 1, i);
        svint32_t row2 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 2, i);
        svint32_t row3 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 3, i);

        auto s0 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, row0), scales0), scaled_ksums0);
        auto s1 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, row1), scales0), scaled_ksums0);
        auto s2 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, row2), scales0), scaled_ksums0);
        auto s3 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, row3), scales0), scaled_ksums0);

        svst1_f32(n_pred, out_ptr + i * p.N + 0 * svcntw() * p.N, s0);
        svst1_f32(n_pred, out_ptr + i * p.N + 1 * svcntw() * p.N, s1);
        svst1_f32(n_pred, out_ptr + i * p.N + 2 * svcntw() * p.N, s2);
        svst1_f32(n_pred, out_ptr + i * p.N + 3 * svcntw() * p.N, s3);
      }
    }
    lhs += svcntb() * p_K_padded;
  }

  // --- Epilogue: 1 M-subtile × 1 N-tile ---
  for (; m < p.M; m += svcntw()) {
    auto rhs_data = static_cast<const int8_t*>(rhs_packed);

    for (size_t n = 0; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;
      svzero_za();

      for (size_t k = 0; k < p.K; k += 4) {
        if constexpr (PF_DIST > 0) {
          svprfb(pg, lhs_data + PF_DIST * svcntb() * 4, SV_PLDL1KEEP);
          svprfb(pg, rhs_data + PF_DIST * svcntb(), SV_PLDL1KEEP);
        }

        auto l0 = svld1_s8(pg, lhs_data);
        auto r0 = svld1_s8(pg, rhs_data);
        svmopa_za32_s8_m(0, pg, pg, l0, r0);

        lhs_data += 4 * svcntb();
        rhs_data += svcntb();
      }

      auto a_scale = svdup_n_f32(qp.a_scale);
      auto w_scales0 = svld1_f32(pg, &qp.w_scales[n]);
      auto scales0 = svmul_f32_x(pg, w_scales0, a_scale);
      auto ksums0 = svld1_f32(pg, &qp.w_ksums[n]);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg, svmul_f32_x(pg, ksums0, zp_f32), a_scale);

      float* out_ptr = out + m * p.N + n;
      svbool_t n_pred = svwhilelt_b32(n, (uint64_t)p.N);

      for (uint32_t i = 0; i < svcntw(); i++) {
        svint32_t row = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i);
        auto s = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, row), scales0), scaled_ksums0);
        svst1_f32(n_pred, out_ptr + i * p.N, s);
      }
    }
    lhs += svcntb();
  }
}

// ---------------------------------------------------------------------------
// Non-streaming wrapper — handles smstart/smstop + ZA lifecycle.
// ---------------------------------------------------------------------------

template <int PF_DIST>
__attribute__((noinline)) __arm_locally_streaming __arm_new("za")
void qc8w_run(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out, const sme::QuantParams& qp) {
  qc8w_kernel<PF_DIST>(p, lhs_packed, rhs_packed, out, qp);
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

template <int PF_DIST>
void BM_qc8w_pf(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  auto pack = sme::gemm_qd8_qc8w_4vlxvl_packing_params();

  std::vector<int8_t> A(M * K), B(K * N);
  std::vector<float> C(M * N);
  auto lhs_packed = std::make_unique<char[]>(sme::packed_size_bytes_s8(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(sme::packed_size_bytes_s8(K, N, pack.rhs));
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s8(B.data(), B.size(), 123);

  std::vector<float> w_scales(N), w_ksums(N);
  fill_random_f32(w_scales.data(), N, 77);
  sme::compute_ksums_s8(B.data(), K, N, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::QuantParams qp{0, 0.05f, w_scales.data(), w_ksums.data()};
  sme::pack_s8(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s8(B.data(), K, N, pack.rhs, rhs_packed.get());

  for (auto _ : state) {
    qc8w_run<PF_DIST>(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double ops = 2.0 * double(M) * double(N) * double(K);
  state.counters["GOP/s"] =
      benchmark::Counter(ops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
  state.counters["pf"] = PF_DIST;
}

#define SIZES \
  ->Args({1024, 1024, 1024})  \
  ->Args({4096, 4096, 4096})  \
  ->Args({1024, 4096, 4096})  \
  ->Unit(benchmark::kMillisecond)

BENCHMARK(BM_qc8w_pf<0>)    SIZES;
BENCHMARK(BM_qc8w_pf<16>)   SIZES;
BENCHMARK(BM_qc8w_pf<32>)   SIZES;
BENCHMARK(BM_qc8w_pf<64>)   SIZES;
BENCHMARK(BM_qc8w_pf<128>)  SIZES;
BENCHMARK(BM_qc8w_pf<256>)  SIZES;
BENCHMARK(BM_qc8w_pf<512>)  SIZES;
BENCHMARK(BM_qc8w_pf<1024>) SIZES;

}  // namespace
