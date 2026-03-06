// 2x2 vs 4x1 tile layout benchmark for qd8_qc8w GEMM kernel.
//
// 4x1: 4 ZA tiles = 4 M-subtiles × 1 N-tile.  5 loads / 4 SMOPAs per K-step.
// 2x2: 4 ZA tiles = 2 M-subtiles × 2 N-tiles. 4 loads / 4 SMOPAs per K-step.

#include "gemm.h"
#include "pack.h"

#include <arm_sme.h>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

// ===========================================================================
// 2x2 kernel — must be at file scope (Apple Clang bug: __arm_locally_streaming
// rejected on static / anonymous-namespace non-template functions).
// ===========================================================================

// 2x2 tile mapping:
//   ZA0 = (m0, n0)    ZA2 = (m0, n1)
//   ZA1 = (m1, n0)    ZA3 = (m1, n1)
//
// K-loop: 4 loads (2 LHS + 2 RHS), 4 SMOPAs.

__attribute__((noinline))
void kernel_2x2(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out, const sme::QuantParams& qp) __arm_streaming __arm_inout("za") {

  auto lhs = static_cast<const int8_t*>(lhs_packed);
  auto rhs_base = static_cast<const int8_t*>(rhs_packed);
  const size_t K_padded = (p.K + 3) & ~size_t(3);
  // Byte stride between consecutive N-tiles in packed RHS.
  const size_t rhs_n_stride = (K_padded / 4) * svcntb();

  svbool_t pg = svptrue_b8();
  size_t m = 0;

  // --- Main body: 2 M-subtiles × 2 N-tiles ---
  for (; m + svcntw() * 2 <= p.M; m += svcntw() * 2) {
    size_t n = 0;
    auto rhs_col = rhs_base;

    for (; n + svcntw() * 2 <= p.N; n += svcntw() * 2) {
      auto lhs_data = lhs;
      auto rhs0 = rhs_col;
      auto rhs1 = rhs_col + rhs_n_stride;
      svzero_za();

      for (size_t k = 0; k < p.K; k += 4) {
        auto l0 = svld1_s8(pg, lhs_data);
        auto l1 = svld1_s8(pg, lhs_data + svcntb());
        auto r0 = svld1_s8(pg, rhs0);
        auto r1 = svld1_s8(pg, rhs1);

        svmopa_za32_s8_m(0, pg, pg, l0, r0);  // ZA0 = (m0, n0)
        svmopa_za32_s8_m(1, pg, pg, l1, r0);  // ZA1 = (m1, n0)
        svmopa_za32_s8_m(2, pg, pg, l0, r1);  // ZA2 = (m0, n1)
        svmopa_za32_s8_m(3, pg, pg, l1, r1);  // ZA3 = (m1, n1)

        lhs_data += 2 * svcntb();
        rhs0 += svcntb();
        rhs1 += svcntb();
      }

      // --- Dequant + store ---
      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));

      // N-tile 0 scales/ksums
      auto ws0 = svld1_f32(pg, &qp.w_scales[n]);
      auto sc0 = svmul_f32_x(pg, ws0, a_scale);
      auto ks0 = svld1_f32(pg, &qp.w_ksums[n]);
      auto sk0 = svmul_f32_x(pg, svmul_f32_x(pg, ks0, zp_f32), a_scale);

      // N-tile 1 scales/ksums
      auto ws1 = svld1_f32(pg, &qp.w_scales[n + svcntw()]);
      auto sc1 = svmul_f32_x(pg, ws1, a_scale);
      auto ks1 = svld1_f32(pg, &qp.w_ksums[n + svcntw()]);
      auto sk1 = svmul_f32_x(pg, svmul_f32_x(pg, ks1, zp_f32), a_scale);

      float* out_ptr = out + m * p.N + n;
      svbool_t np0 = svwhilelt_b32(n, (uint64_t)p.N);
      svbool_t np1 = svwhilelt_b32(n + svcntw(), (uint64_t)p.N);

      for (uint32_t i = 0; i < svcntw(); i++) {
        // ZA0 row i -> (m0+i, n0), ZA2 row i -> (m0+i, n1)
        auto r0 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i);
        auto r2 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 2, i);
        auto s0 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, r0), sc0), sk0);
        auto s2 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, r2), sc1), sk1);
        svst1_f32(np0, out_ptr + i * p.N, s0);
        svst1_f32(np1, out_ptr + i * p.N + svcntw(), s2);

        // ZA1 row i -> (m1+i, n0), ZA3 row i -> (m1+i, n1)
        auto r1 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 1, i);
        auto r3 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 3, i);
        auto s1 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, r1), sc0), sk0);
        auto s3 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, r3), sc1), sk1);
        svst1_f32(np0, out_ptr + (i + svcntw()) * p.N, s1);
        svst1_f32(np1, out_ptr + (i + svcntw()) * p.N + svcntw(), s3);
      }

      rhs_col += 2 * rhs_n_stride;
    }

    // --- N-tail: 2x1 (odd trailing N-tile) ---
    if (n < p.N) {
      auto lhs_data = lhs;
      auto rhs0 = rhs_col;
      svzero_za();

      for (size_t k = 0; k < p.K; k += 4) {
        auto l0 = svld1_s8(pg, lhs_data);
        auto l1 = svld1_s8(pg, lhs_data + svcntb());
        auto r0 = svld1_s8(pg, rhs0);

        svmopa_za32_s8_m(0, pg, pg, l0, r0);
        svmopa_za32_s8_m(1, pg, pg, l1, r0);

        lhs_data += 2 * svcntb();
        rhs0 += svcntb();
      }

      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto ws0 = svld1_f32(pg, &qp.w_scales[n]);
      auto sc0 = svmul_f32_x(pg, ws0, a_scale);
      auto ks0 = svld1_f32(pg, &qp.w_ksums[n]);
      auto sk0 = svmul_f32_x(pg, svmul_f32_x(pg, ks0, zp_f32), a_scale);

      float* out_ptr = out + m * p.N + n;
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);

      for (uint32_t i = 0; i < svcntw(); i++) {
        auto r0 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i);
        auto r1 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 1, i);
        auto s0 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, r0), sc0), sk0);
        auto s1 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, r1), sc0), sk0);
        svst1_f32(np, out_ptr + i * p.N, s0);
        svst1_f32(np, out_ptr + (i + svcntw()) * p.N, s1);
      }
    }

    lhs += 2 * svcntb() * (K_padded / 4);
  }

  // --- M epilogue: 1x1 for remaining rows ---
  for (; m < p.M; m += svcntw()) {
    auto rhs_col = rhs_base;

    for (size_t n = 0; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;
      auto rhs0 = rhs_col;
      svzero_za();

      for (size_t k = 0; k < p.K; k += 4) {
        auto l0 = svld1_s8(pg, lhs_data);
        auto r0 = svld1_s8(pg, rhs0);
        svmopa_za32_s8_m(0, pg, pg, l0, r0);

        // LHS is packed with tile_rows=2*vl: skip both subtiles.
        lhs_data += 2 * svcntb();
        rhs0 += svcntb();
      }

      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto ws0 = svld1_f32(pg, &qp.w_scales[n]);
      auto sc0 = svmul_f32_x(pg, ws0, a_scale);
      auto ks0 = svld1_f32(pg, &qp.w_ksums[n]);
      auto sk0 = svmul_f32_x(pg, svmul_f32_x(pg, ks0, zp_f32), a_scale);

      float* out_ptr = out + m * p.N + n;
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);

      for (uint32_t i = 0; i < svcntw(); i++) {
        auto row = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i);
        auto s = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, row), sc0), sk0);
        svst1_f32(np, out_ptr + i * p.N, s);
      }

      rhs_col += rhs_n_stride;
    }

    lhs += svcntb();
  }
}

// Non-streaming wrapper.
__attribute__((noinline)) __arm_locally_streaming __arm_new("za")
void run_2x2(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out, const sme::QuantParams& qp) {
  kernel_2x2(p, lhs_packed, rhs_packed, out, qp);
}

// ===========================================================================
// Benchmark harness
// ===========================================================================

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

// --- 4x1 baseline (library kernel) ------------------------------------------

void BM_qd8_4x1(benchmark::State& state) {
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
    sme::gemm_qd8p_qc8wp_f32_4vlxvl(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * double(M) * double(N) * double(K);
  state.counters["GFLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
}

// --- 2x2 benchmark ----------------------------------------------------------

void BM_qd8_2x2(benchmark::State& state) {
  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  // 2x2 packing: LHS tile_rows = vl*2 (instead of vl*4).
  // RHS packing is unchanged.
  auto pack = sme::gemm_qd8_qc8w_4vlxvl_packing_params();
  pack.lhs.tile_rows /= 2;  // 2 subtiles instead of 4

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
    run_2x2(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  double flops = 2.0 * double(M) * double(N) * double(K);
  state.counters["GFLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate,
                         benchmark::Counter::kIs1000);
}

// --- Registration ------------------------------------------------------------

#define SIZES \
  ->Args({1024, 1024, 1024})  \
  ->Args({4096, 4096, 4096})  \
  ->Args({1024, 4096, 4096})  \
  ->Args({128, 131072, 4096})  \
  ->Unit(benchmark::kMillisecond)

BENCHMARK(BM_qd8_4x1) SIZES;
BENCHMARK(BM_qd8_2x2) SIZES;

}  // namespace
