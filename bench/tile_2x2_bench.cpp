// 2x2 vs 4x1 tile layout benchmark for all GEMM dtypes.
//
// 4x1: 4 ZA tiles = 4 M-subtiles × 1 N-tile.  5 loads / 4 MOPAs per K-step.
// 2x2: 4 ZA tiles = 2 M-subtiles × 2 N-tiles. 4 loads / 4 MOPAs per K-step.
//
// Each benchmark verifies that 2x2 output matches 4x1 before timing.

#include "gemm.h"
#include "pack.h"

#include <arm_sme.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

// ===========================================================================
// 2x2 kernels — file scope (Apple Clang rejects __arm_locally_streaming on
// static / anonymous-namespace non-template functions).
// ===========================================================================

// ---------- f32 (rank-1, K-step=1) ------------------------------------------

__attribute__((noinline))
void kernel_2x2_f32(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out) __arm_streaming __arm_inout("za") {

  const auto* lhs_base = static_cast<const float*>(lhs_packed);
  const auto* rhs_base = static_cast<const float*>(rhs_packed);
  const uint64_t vl = svcntw();
  const svbool_t pg = svptrue_b32();

  uint64_t m = 0;
  for (; m + vl * 2 <= p.M; m += vl * 2) {
    const float* lhs_m = lhs_base + (m / (vl * 2)) * p.K * vl * 2;

    uint64_t n = 0;
    for (; n + vl * 2 <= p.N; n += vl * 2) {
      const float* rhs_n0 = rhs_base + (n / vl) * p.K * vl;
      const float* rhs_n1 = rhs_n0 + p.K * vl;

      svzero_za();
      const float* ld = lhs_m;
      const float* r0 = rhs_n0;
      const float* r1 = rhs_n1;

      for (uint64_t k = 0; k < p.K; k++) {
        auto l0 = svld1_f32(pg, ld);
        auto l1 = svld1_f32(pg, ld + vl);
        auto rv0 = svld1_f32(pg, r0);
        auto rv1 = svld1_f32(pg, r1);

        svmopa_za32_f32_m(0, pg, pg, l0, rv0);
        svmopa_za32_f32_m(1, pg, pg, l1, rv0);
        svmopa_za32_f32_m(2, pg, pg, l0, rv1);
        svmopa_za32_f32_m(3, pg, pg, l1, rv1);

        ld += vl * 2;
        r0 += vl;
        r1 += vl;
      }

      svbool_t np0 = svwhilelt_b32(n, (uint64_t)p.N);
      svbool_t np1 = svwhilelt_b32(n + vl, (uint64_t)p.N);
      float* op = out + m * p.N + n;
      for (uint32_t i = 0; i < vl; i++) {
        svst1_hor_za32(0, i, np0, op + i * p.N);
        svst1_hor_za32(2, i, np1, op + i * p.N + vl);
        svst1_hor_za32(1, i, np0, op + (i + vl) * p.N);
        svst1_hor_za32(3, i, np1, op + (i + vl) * p.N + vl);
      }
    }

    // N-tail: 2x1
    if (n < p.N) {
      const float* rhs_n0 = rhs_base + (n / vl) * p.K * vl;
      svzero_za();
      const float* ld = lhs_m;
      const float* r0 = rhs_n0;
      for (uint64_t k = 0; k < p.K; k++) {
        svmopa_za32_f32_m(0, pg, pg, svld1_f32(pg, ld), svld1_f32(pg, r0));
        svmopa_za32_f32_m(1, pg, pg, svld1_f32(pg, ld + vl), svld1_f32(pg, r0));
        ld += vl * 2;
        r0 += vl;
      }
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      float* op = out + m * p.N + n;
      for (uint32_t i = 0; i < vl; i++) {
        svst1_hor_za32(0, i, np, op + i * p.N);
        svst1_hor_za32(1, i, np, op + (i + vl) * p.N);
      }
    }
  }

  // M epilogue: 1x1
  for (; m < p.M; m += vl) {
    const float* lhs_m = lhs_base + (m / (vl * 2)) * p.K * vl * 2;
    const uint64_t rows = std::min(vl, p.M - m);

    for (uint64_t n = 0; n < p.N; n += vl) {
      const float* rhs_n = rhs_base + (n / vl) * p.K * vl;
      svzero_za();
      const float* ld = lhs_m;
      const float* r0 = rhs_n;
      svbool_t mp = svwhilelt_b32(m, (uint64_t)p.M);
      for (uint64_t k = 0; k < p.K; k++) {
        svmopa_za32_f32_m(0, mp, pg, svld1_f32(mp, ld), svld1_f32(pg, r0));
        ld += vl * 2;
        r0 += vl;
      }
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      float* op = out + m * p.N + n;
      for (uint32_t i = 0; i < rows; i++)
        svst1_hor_za32(0, i, np, op + i * p.N);
    }
  }
}

__attribute__((noinline)) __arm_locally_streaming __arm_new("za")
void run_2x2_f32(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out) {
  kernel_2x2_f32(p, lhs_packed, rhs_packed, out);
}

// ---------- f16 (rank-2 widening, K-step=2) ---------------------------------

__attribute__((noinline))
void kernel_2x2_f16(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    _Float16* out) __arm_streaming __arm_inout("za") {

  const auto* lhs_base = static_cast<const float16_t*>(lhs_packed);
  const auto* rhs_base = static_cast<const float16_t*>(rhs_packed);
  const uint64_t vl = svcnth();        // f16 elements per vector
  const uint64_t vl32 = svcntw();      // f32 elements per vector
  const uint64_t K_pad = (p.K + 1) & ~1ULL;
  const uint64_t rhs_n_stride = K_pad * vl32;  // f16 elements between N-tiles

  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t fz = svdup_n_f32(0);

  uint64_t m = 0;
  for (; m + vl32 * 2 <= p.M; m += vl32 * 2) {
    const float16_t* lhs_m = lhs_base + (m / (vl32 * 2)) * K_pad * vl32 * 2;

    uint64_t n = 0;
    const float16_t* rhs_col = rhs_base;
    for (; n + vl32 * 2 <= p.N; n += vl32 * 2) {
      svzero_za();
      const float16_t* ld = lhs_m;
      const float16_t* r0 = rhs_col;
      const float16_t* r1 = rhs_col + rhs_n_stride;

      for (uint64_t k = 0; k < p.K; k += 2) {
        auto l0 = svld1_f16(pg16, ld);
        auto l1 = svld1_f16(pg16, ld + vl);
        auto rv0 = svld1_f16(pg16, r0);
        auto rv1 = svld1_f16(pg16, r1);

        svmopa_za32_f16_m(0, pg16, pg16, l0, rv0);
        svmopa_za32_f16_m(1, pg16, pg16, l1, rv0);
        svmopa_za32_f16_m(2, pg16, pg16, l0, rv1);
        svmopa_za32_f16_m(3, pg16, pg16, l1, rv1);

        ld += vl * 2;
        r0 += vl;
        r1 += vl;
      }

      auto* dst = reinterpret_cast<int16_t*>(out);
      svbool_t np0 = svwhilelt_b32(n, (uint64_t)p.N);
      svbool_t np1 = svwhilelt_b32(n + vl32, (uint64_t)p.N);
      for (uint32_t i = 0; i < vl32; i++) {
        auto z0 = svread_hor_za32_f32_m(fz, pg32, 0, i);
        auto z2 = svread_hor_za32_f32_m(fz, pg32, 2, i);
        auto z1 = svread_hor_za32_f32_m(fz, pg32, 1, i);
        auto z3 = svread_hor_za32_f32_m(fz, pg32, 3, i);
        svst1h_s32(np0, dst + (m + i) * p.N + n,
                   svreinterpret_s32_f16(svcvt_f16_f32_x(pg32, z0)));
        svst1h_s32(np1, dst + (m + i) * p.N + n + vl32,
                   svreinterpret_s32_f16(svcvt_f16_f32_x(pg32, z2)));
        svst1h_s32(np0, dst + (m + vl32 + i) * p.N + n,
                   svreinterpret_s32_f16(svcvt_f16_f32_x(pg32, z1)));
        svst1h_s32(np1, dst + (m + vl32 + i) * p.N + n + vl32,
                   svreinterpret_s32_f16(svcvt_f16_f32_x(pg32, z3)));
      }
      rhs_col += 2 * rhs_n_stride;
    }

    // N-tail: 2x1
    if (n < p.N) {
      svzero_za();
      const float16_t* ld = lhs_m;
      const float16_t* r0 = rhs_col;
      for (uint64_t k = 0; k < p.K; k += 2) {
        auto l0 = svld1_f16(pg16, ld);
        auto l1 = svld1_f16(pg16, ld + vl);
        auto rv0 = svld1_f16(pg16, r0);
        svmopa_za32_f16_m(0, pg16, pg16, l0, rv0);
        svmopa_za32_f16_m(1, pg16, pg16, l1, rv0);
        ld += vl * 2;
        r0 += vl;
      }
      auto* dst = reinterpret_cast<int16_t*>(out);
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      for (uint32_t i = 0; i < vl32; i++) {
        svst1h_s32(np, dst + (m + i) * p.N + n,
                   svreinterpret_s32_f16(svcvt_f16_f32_x(pg32,
                       svread_hor_za32_f32_m(fz, pg32, 0, i))));
        svst1h_s32(np, dst + (m + vl32 + i) * p.N + n,
                   svreinterpret_s32_f16(svcvt_f16_f32_x(pg32,
                       svread_hor_za32_f32_m(fz, pg32, 1, i))));
      }
    }
  }

  // M epilogue: 1x1
  for (; m < p.M; m += vl32) {
    const float16_t* lhs_m = lhs_base + (m / (vl32 * 2)) * K_pad * vl32 * 2;
    const uint64_t rows = std::min(vl32, p.M - m);
    const float16_t* rhs_col = rhs_base;

    for (uint64_t n = 0; n < p.N; n += vl32) {
      svzero_za();
      const float16_t* ld = lhs_m;
      const float16_t* r0 = rhs_col;
      for (uint64_t k = 0; k < p.K; k += 2) {
        svmopa_za32_f16_m(0, pg16, pg16, svld1_f16(pg16, ld), svld1_f16(pg16, r0));
        ld += vl * 2;
        r0 += vl;
      }
      auto* dst = reinterpret_cast<int16_t*>(out);
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      for (uint32_t i = 0; i < rows; i++)
        svst1h_s32(np, dst + (m + i) * p.N + n,
                   svreinterpret_s32_f16(svcvt_f16_f32_x(pg32,
                       svread_hor_za32_f32_m(fz, pg32, 0, i))));
      rhs_col += rhs_n_stride;
    }
  }
}

__attribute__((noinline)) __arm_locally_streaming __arm_new("za")
void run_2x2_f16(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    _Float16* out) {
  kernel_2x2_f16(p, lhs_packed, rhs_packed, out);
}

// ---------- bf16 (rank-2 widening, K-step=2) --------------------------------

__attribute__((noinline))
void kernel_2x2_bf16(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    __bf16* out) __arm_streaming __arm_inout("za") {

  const auto* lhs_base = static_cast<const bfloat16_t*>(lhs_packed);
  const auto* rhs_base = static_cast<const bfloat16_t*>(rhs_packed);
  const uint64_t vl = svcnth();
  const uint64_t vl32 = svcntw();
  const uint64_t K_pad = (p.K + 1) & ~1ULL;
  const uint64_t rhs_n_stride = K_pad * vl32;

  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t fz = svdup_n_f32(0);

  uint64_t m = 0;
  for (; m + vl32 * 2 <= p.M; m += vl32 * 2) {
    const bfloat16_t* lhs_m = lhs_base + (m / (vl32 * 2)) * K_pad * vl32 * 2;

    uint64_t n = 0;
    const bfloat16_t* rhs_col = rhs_base;
    for (; n + vl32 * 2 <= p.N; n += vl32 * 2) {
      svzero_za();
      const bfloat16_t* ld = lhs_m;
      const bfloat16_t* r0 = rhs_col;
      const bfloat16_t* r1 = rhs_col + rhs_n_stride;

      for (uint64_t k = 0; k < p.K; k += 2) {
        auto l0 = svld1_bf16(pg16, ld);
        auto l1 = svld1_bf16(pg16, ld + vl);
        auto rv0 = svld1_bf16(pg16, r0);
        auto rv1 = svld1_bf16(pg16, r1);

        svmopa_za32_bf16_m(0, pg16, pg16, l0, rv0);
        svmopa_za32_bf16_m(1, pg16, pg16, l1, rv0);
        svmopa_za32_bf16_m(2, pg16, pg16, l0, rv1);
        svmopa_za32_bf16_m(3, pg16, pg16, l1, rv1);

        ld += vl * 2;
        r0 += vl;
        r1 += vl;
      }

      auto* dst = reinterpret_cast<int16_t*>(out);
      svbool_t np0 = svwhilelt_b32(n, (uint64_t)p.N);
      svbool_t np1 = svwhilelt_b32(n + vl32, (uint64_t)p.N);
      for (uint32_t i = 0; i < vl32; i++) {
        auto z0 = svread_hor_za32_f32_m(fz, pg32, 0, i);
        auto z2 = svread_hor_za32_f32_m(fz, pg32, 2, i);
        auto z1 = svread_hor_za32_f32_m(fz, pg32, 1, i);
        auto z3 = svread_hor_za32_f32_m(fz, pg32, 3, i);
        svst1h_s32(np0, dst + (m + i) * p.N + n,
                   svreinterpret_s32_bf16(svcvt_bf16_f32_x(pg32, z0)));
        svst1h_s32(np1, dst + (m + i) * p.N + n + vl32,
                   svreinterpret_s32_bf16(svcvt_bf16_f32_x(pg32, z2)));
        svst1h_s32(np0, dst + (m + vl32 + i) * p.N + n,
                   svreinterpret_s32_bf16(svcvt_bf16_f32_x(pg32, z1)));
        svst1h_s32(np1, dst + (m + vl32 + i) * p.N + n + vl32,
                   svreinterpret_s32_bf16(svcvt_bf16_f32_x(pg32, z3)));
      }
      rhs_col += 2 * rhs_n_stride;
    }

    if (n < p.N) {
      svzero_za();
      const bfloat16_t* ld = lhs_m;
      const bfloat16_t* r0 = rhs_col;
      for (uint64_t k = 0; k < p.K; k += 2) {
        auto l0 = svld1_bf16(pg16, ld);
        auto l1 = svld1_bf16(pg16, ld + vl);
        auto rv0 = svld1_bf16(pg16, r0);
        svmopa_za32_bf16_m(0, pg16, pg16, l0, rv0);
        svmopa_za32_bf16_m(1, pg16, pg16, l1, rv0);
        ld += vl * 2;
        r0 += vl;
      }
      auto* dst = reinterpret_cast<int16_t*>(out);
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      for (uint32_t i = 0; i < vl32; i++) {
        svst1h_s32(np, dst + (m + i) * p.N + n,
                   svreinterpret_s32_bf16(svcvt_bf16_f32_x(pg32,
                       svread_hor_za32_f32_m(fz, pg32, 0, i))));
        svst1h_s32(np, dst + (m + vl32 + i) * p.N + n,
                   svreinterpret_s32_bf16(svcvt_bf16_f32_x(pg32,
                       svread_hor_za32_f32_m(fz, pg32, 1, i))));
      }
    }
  }

  for (; m < p.M; m += vl32) {
    const bfloat16_t* lhs_m = lhs_base + (m / (vl32 * 2)) * K_pad * vl32 * 2;
    const uint64_t rows = std::min(vl32, p.M - m);
    const bfloat16_t* rhs_col = rhs_base;

    for (uint64_t n = 0; n < p.N; n += vl32) {
      svzero_za();
      const bfloat16_t* ld = lhs_m;
      const bfloat16_t* r0 = rhs_col;
      for (uint64_t k = 0; k < p.K; k += 2) {
        svmopa_za32_bf16_m(0, pg16, pg16, svld1_bf16(pg16, ld), svld1_bf16(pg16, r0));
        ld += vl * 2;
        r0 += vl;
      }
      auto* dst = reinterpret_cast<int16_t*>(out);
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      for (uint32_t i = 0; i < rows; i++)
        svst1h_s32(np, dst + (m + i) * p.N + n,
                   svreinterpret_s32_bf16(svcvt_bf16_f32_x(pg32,
                       svread_hor_za32_f32_m(fz, pg32, 0, i))));
      rhs_col += rhs_n_stride;
    }
  }
}

__attribute__((noinline)) __arm_locally_streaming __arm_new("za")
void run_2x2_bf16(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    __bf16* out) {
  kernel_2x2_bf16(p, lhs_packed, rhs_packed, out);
}

// ---------- qd8_qc8w (rank-4, K-step=4) ------------------------------------

__attribute__((noinline))
void kernel_2x2_qd8(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out, const sme::QuantParams& qp) __arm_streaming __arm_inout("za") {

  auto lhs = static_cast<const int8_t*>(lhs_packed);
  auto rhs_base = static_cast<const int8_t*>(rhs_packed);
  const size_t K_pad = (p.K + 3) & ~size_t(3);
  const size_t rhs_n_stride = (K_pad / 4) * svcntb();

  svbool_t pg = svptrue_b8();
  size_t m = 0;

  for (; m + svcntw() * 2 <= p.M; m += svcntw() * 2) {
    size_t n = 0;
    auto rhs_col = rhs_base;

    for (; n + svcntw() * 2 <= p.N; n += svcntw() * 2) {
      auto ld = lhs;
      auto r0 = rhs_col;
      auto r1 = rhs_col + rhs_n_stride;
      svzero_za();

      for (size_t k = 0; k < p.K; k += 4) {
        auto l0 = svld1_s8(pg, ld);
        auto l1 = svld1_s8(pg, ld + svcntb());
        auto rv0 = svld1_s8(pg, r0);
        auto rv1 = svld1_s8(pg, r1);
        svmopa_za32_s8_m(0, pg, pg, l0, rv0);
        svmopa_za32_s8_m(1, pg, pg, l1, rv0);
        svmopa_za32_s8_m(2, pg, pg, l0, rv1);
        svmopa_za32_s8_m(3, pg, pg, l1, rv1);
        ld += 2 * svcntb();
        r0 += svcntb();
        r1 += svcntb();
      }

      auto a_sc = svdup_n_f32(qp.a_scale);
      auto zp = svcvt_f32_s32_x(pg, svdup_n_s32((int32_t)qp.a_zero_point));
      auto ws0 = svmul_f32_x(pg, svld1_f32(pg, &qp.w_scales[n]), a_sc);
      auto sk0 = svmul_f32_x(pg, svmul_f32_x(pg, svld1_f32(pg, &qp.w_ksums[n]), zp), a_sc);
      auto ws1 = svmul_f32_x(pg, svld1_f32(pg, &qp.w_scales[n + svcntw()]), a_sc);
      auto sk1 = svmul_f32_x(pg, svmul_f32_x(pg, svld1_f32(pg, &qp.w_ksums[n + svcntw()]), zp), a_sc);

      float* op = out + m * p.N + n;
      svbool_t np0 = svwhilelt_b32(n, (uint64_t)p.N);
      svbool_t np1 = svwhilelt_b32(n + svcntw(), (uint64_t)p.N);

      for (uint32_t i = 0; i < svcntw(); i++) {
        auto v0 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i)), ws0), sk0);
        auto v2 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 2, i)), ws1), sk1);
        svst1_f32(np0, op + i * p.N, v0);
        svst1_f32(np1, op + i * p.N + svcntw(), v2);

        auto v1 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 1, i)), ws0), sk0);
        auto v3 = svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 3, i)), ws1), sk1);
        svst1_f32(np0, op + (i + svcntw()) * p.N, v1);
        svst1_f32(np1, op + (i + svcntw()) * p.N + svcntw(), v3);
      }
      rhs_col += 2 * rhs_n_stride;
    }

    if (n < p.N) {
      auto ld = lhs;
      auto r0 = rhs_col;
      svzero_za();
      for (size_t k = 0; k < p.K; k += 4) {
        svmopa_za32_s8_m(0, pg, pg, svld1_s8(pg, ld), svld1_s8(pg, r0));
        svmopa_za32_s8_m(1, pg, pg, svld1_s8(pg, ld + svcntb()), svld1_s8(pg, r0));
        ld += 2 * svcntb();
        r0 += svcntb();
      }
      auto a_sc = svdup_n_f32(qp.a_scale);
      auto zp = svcvt_f32_s32_x(pg, svdup_n_s32((int32_t)qp.a_zero_point));
      auto ws0 = svmul_f32_x(pg, svld1_f32(pg, &qp.w_scales[n]), a_sc);
      auto sk0 = svmul_f32_x(pg, svmul_f32_x(pg, svld1_f32(pg, &qp.w_ksums[n]), zp), a_sc);
      float* op = out + m * p.N + n;
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      for (uint32_t i = 0; i < svcntw(); i++) {
        svst1_f32(np, op + i * p.N, svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i)), ws0), sk0));
        svst1_f32(np, op + (i + svcntw()) * p.N, svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 1, i)), ws0), sk0));
      }
    }
    lhs += 2 * svcntb() * (K_pad / 4);
  }

  for (; m < p.M; m += svcntw()) {
    auto rhs_col = rhs_base;
    for (size_t n = 0; n < p.N; n += svcntw()) {
      auto ld = lhs;
      auto r0 = rhs_col;
      svzero_za();
      for (size_t k = 0; k < p.K; k += 4) {
        svmopa_za32_s8_m(0, pg, pg, svld1_s8(pg, ld), svld1_s8(pg, r0));
        ld += 2 * svcntb();
        r0 += svcntb();
      }
      auto a_sc = svdup_n_f32(qp.a_scale);
      auto zp = svcvt_f32_s32_x(pg, svdup_n_s32((int32_t)qp.a_zero_point));
      auto ws0 = svmul_f32_x(pg, svld1_f32(pg, &qp.w_scales[n]), a_sc);
      auto sk0 = svmul_f32_x(pg, svmul_f32_x(pg, svld1_f32(pg, &qp.w_ksums[n]), zp), a_sc);
      float* op = out + m * p.N + n;
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      for (uint32_t i = 0; i < svcntw(); i++)
        svst1_f32(np, op + i * p.N, svsub_f32_x(pg, svmul_f32_x(pg, svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i)), ws0), sk0));
      rhs_col += rhs_n_stride;
    }
    lhs += svcntb();
  }
}

__attribute__((noinline)) __arm_locally_streaming __arm_new("za")
void run_2x2_qd8(
    const sme::GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out, const sme::QuantParams& qp) {
  kernel_2x2_qd8(p, lhs_packed, rhs_packed, out, qp);
}

// ===========================================================================
// Benchmark harness
// ===========================================================================

namespace {

void fill_random_f32(float* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

void fill_random_f16(_Float16* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<_Float16>(dist(rng));
}

void fill_random_bf16(__bf16* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<__bf16>(dist(rng));
}

void fill_random_s8(int8_t* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-128, 127);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<int8_t>(dist(rng));
}

void fill_random_f32_pos(float* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.01f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

// Verify: returns max absolute diff. Prints first mismatch.
template <typename T>
float verify(const T* ref, const T* tst, size_t n, const char* label) {
  float maxd = 0;
  size_t first_bad = ~size_t(0);
  for (size_t i = 0; i < n; i++) {
    float d = std::abs(static_cast<float>(ref[i]) - static_cast<float>(tst[i]));
    if (d > maxd) {
      maxd = d;
      if (first_bad == ~size_t(0)) first_bad = i;
    }
  }
  if (maxd > 0)
    fprintf(stderr, "[%s] max_diff=%.6g @ idx %zu (ref=%.6g tst=%.6g)\n",
            label, maxd, first_bad,
            static_cast<float>(ref[first_bad]),
            static_cast<float>(tst[first_bad]));
  return maxd;
}

// ---------- f32 benchmarks ---------------------------------------------------

void BM_f32_4x1(benchmark::State& state) {
  const size_t M = state.range(0), N = state.range(1), K = state.range(2);
  auto pk = sme::gemm_f32_4vlxvl_packing_params();

  std::vector<float> A(M*K), B(K*N), C(M*N);
  auto lp = std::make_unique<char[]>(sme::packed_size_bytes_f32(M, K, pk.lhs));
  auto rp = std::make_unique<char[]>(sme::packed_size_bytes_f32(K, N, pk.rhs));
  fill_random_f32(A.data(), A.size(), 42);
  fill_random_f32(B.data(), B.size(), 123);
  sme::pack_f32(A.data(), M, K, pk.lhs, lp.get());
  sme::pack_f32(B.data(), K, N, pk.rhs, rp.get());
  sme::GemmParams p{M, N, K};

  for (auto _ : state) {
    sme::gemm_f32p_f32p_f32_4vlxvl(p, lp.get(), rp.get(), C.data());
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOP/s"] = benchmark::Counter(
      2.0*M*N*K, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
}

void BM_f32_2x2(benchmark::State& state) {
  const size_t M = state.range(0), N = state.range(1), K = state.range(2);
  auto pk4 = sme::gemm_f32_4vlxvl_packing_params();
  auto pk2 = pk4;
  pk2.lhs.tile_rows /= 2;

  std::vector<float> A(M*K), B(K*N), C_ref(M*N), C_2x2(M*N);
  fill_random_f32(A.data(), A.size(), 42);
  fill_random_f32(B.data(), B.size(), 123);

  auto lp4 = std::make_unique<char[]>(sme::packed_size_bytes_f32(M, K, pk4.lhs));
  auto lp2 = std::make_unique<char[]>(sme::packed_size_bytes_f32(M, K, pk2.lhs));
  auto rp  = std::make_unique<char[]>(sme::packed_size_bytes_f32(K, N, pk4.rhs));
  sme::pack_f32(A.data(), M, K, pk4.lhs, lp4.get());
  sme::pack_f32(A.data(), M, K, pk2.lhs, lp2.get());
  sme::pack_f32(B.data(), K, N, pk4.rhs, rp.get());
  sme::GemmParams p{M, N, K};

  // Verify
  sme::gemm_f32p_f32p_f32_4vlxvl(p, lp4.get(), rp.get(), C_ref.data());
  run_2x2_f32(p, lp2.get(), rp.get(), C_2x2.data());
  float d = verify(C_ref.data(), C_2x2.data(), M*N, "f32");
  if (d > 1e-3f) { state.SkipWithError("f32 2x2 verification failed"); return; }

  for (auto _ : state) {
    run_2x2_f32(p, lp2.get(), rp.get(), C_2x2.data());
    benchmark::DoNotOptimize(C_2x2.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOP/s"] = benchmark::Counter(
      2.0*M*N*K, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
}

// ---------- f16 benchmarks ---------------------------------------------------

void BM_f16_4x1(benchmark::State& state) {
  const size_t M = state.range(0), N = state.range(1), K = state.range(2);
  auto pk = sme::gemm_f16_4vlxvl_packing_params();

  std::vector<_Float16> A(M*K), B(K*N), C(M*N);
  auto lp = std::make_unique<char[]>(sme::packed_size_bytes_f16(M, K, pk.lhs));
  auto rp = std::make_unique<char[]>(sme::packed_size_bytes_f16(K, N, pk.rhs));
  fill_random_f16(A.data(), A.size(), 42);
  fill_random_f16(B.data(), B.size(), 123);
  sme::pack_f16(A.data(), M, K, pk.lhs, lp.get());
  sme::pack_f16(B.data(), K, N, pk.rhs, rp.get());
  sme::GemmParams p{M, N, K};

  for (auto _ : state) {
    sme::gemm_f16p_f16p_f16_4vlxvl(p, lp.get(), rp.get(), C.data());
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOP/s"] = benchmark::Counter(
      2.0*M*N*K, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
}

void BM_f16_2x2(benchmark::State& state) {
  const size_t M = state.range(0), N = state.range(1), K = state.range(2);
  auto pk4 = sme::gemm_f16_4vlxvl_packing_params();
  auto pk2 = pk4;
  pk2.lhs.tile_rows /= 2;

  std::vector<_Float16> A(M*K), B(K*N), C_ref(M*N), C_2x2(M*N);
  fill_random_f16(A.data(), A.size(), 42);
  fill_random_f16(B.data(), B.size(), 123);

  auto lp4 = std::make_unique<char[]>(sme::packed_size_bytes_f16(M, K, pk4.lhs));
  auto lp2 = std::make_unique<char[]>(sme::packed_size_bytes_f16(M, K, pk2.lhs));
  auto rp  = std::make_unique<char[]>(sme::packed_size_bytes_f16(K, N, pk4.rhs));
  sme::pack_f16(A.data(), M, K, pk4.lhs, lp4.get());
  sme::pack_f16(A.data(), M, K, pk2.lhs, lp2.get());
  sme::pack_f16(B.data(), K, N, pk4.rhs, rp.get());
  sme::GemmParams p{M, N, K};

  sme::gemm_f16p_f16p_f16_4vlxvl(p, lp4.get(), rp.get(), C_ref.data());
  run_2x2_f16(p, lp2.get(), rp.get(), C_2x2.data());
  float d = verify(C_ref.data(), C_2x2.data(), M*N, "f16");
  if (d > 1e-1f) { state.SkipWithError("f16 2x2 verification failed"); return; }

  for (auto _ : state) {
    run_2x2_f16(p, lp2.get(), rp.get(), C_2x2.data());
    benchmark::DoNotOptimize(C_2x2.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOP/s"] = benchmark::Counter(
      2.0*M*N*K, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
}

// ---------- bf16 benchmarks --------------------------------------------------

void BM_bf16_4x1(benchmark::State& state) {
  const size_t M = state.range(0), N = state.range(1), K = state.range(2);
  auto pk = sme::gemm_bf16_4vlxvl_packing_params();

  std::vector<__bf16> A(M*K), B(K*N), C(M*N);
  auto lp = std::make_unique<char[]>(sme::packed_size_bytes_bf16(M, K, pk.lhs));
  auto rp = std::make_unique<char[]>(sme::packed_size_bytes_bf16(K, N, pk.rhs));
  fill_random_bf16(A.data(), A.size(), 42);
  fill_random_bf16(B.data(), B.size(), 123);
  sme::pack_bf16(A.data(), M, K, pk.lhs, lp.get());
  sme::pack_bf16(B.data(), K, N, pk.rhs, rp.get());
  sme::GemmParams p{M, N, K};

  for (auto _ : state) {
    sme::gemm_bf16p_bf16p_bf16_4vlxvl(p, lp.get(), rp.get(), C.data());
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOP/s"] = benchmark::Counter(
      2.0*M*N*K, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
}

void BM_bf16_2x2(benchmark::State& state) {
  const size_t M = state.range(0), N = state.range(1), K = state.range(2);
  auto pk4 = sme::gemm_bf16_4vlxvl_packing_params();
  auto pk2 = pk4;
  pk2.lhs.tile_rows /= 2;

  std::vector<__bf16> A(M*K), B(K*N), C_ref(M*N), C_2x2(M*N);
  fill_random_bf16(A.data(), A.size(), 42);
  fill_random_bf16(B.data(), B.size(), 123);

  auto lp4 = std::make_unique<char[]>(sme::packed_size_bytes_bf16(M, K, pk4.lhs));
  auto lp2 = std::make_unique<char[]>(sme::packed_size_bytes_bf16(M, K, pk2.lhs));
  auto rp  = std::make_unique<char[]>(sme::packed_size_bytes_bf16(K, N, pk4.rhs));
  sme::pack_bf16(A.data(), M, K, pk4.lhs, lp4.get());
  sme::pack_bf16(A.data(), M, K, pk2.lhs, lp2.get());
  sme::pack_bf16(B.data(), K, N, pk4.rhs, rp.get());
  sme::GemmParams p{M, N, K};

  sme::gemm_bf16p_bf16p_bf16_4vlxvl(p, lp4.get(), rp.get(), C_ref.data());
  run_2x2_bf16(p, lp2.get(), rp.get(), C_2x2.data());
  float d = verify(C_ref.data(), C_2x2.data(), M*N, "bf16");
  if (d > 1e-1f) { state.SkipWithError("bf16 2x2 verification failed"); return; }

  for (auto _ : state) {
    run_2x2_bf16(p, lp2.get(), rp.get(), C_2x2.data());
    benchmark::DoNotOptimize(C_2x2.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOP/s"] = benchmark::Counter(
      2.0*M*N*K, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
}

// ---------- qd8 benchmarks ---------------------------------------------------

void BM_qd8_4x1(benchmark::State& state) {
  const size_t M = state.range(0), N = state.range(1), K = state.range(2);
  auto pk = sme::gemm_qd8_qc8w_4vlxvl_packing_params();

  std::vector<int8_t> A(M*K), B(K*N);
  std::vector<float> C(M*N), w_scales(N), w_ksums(N);
  auto lp = std::make_unique<char[]>(sme::packed_size_bytes_s8(M, K, pk.lhs));
  auto rp = std::make_unique<char[]>(sme::packed_size_bytes_s8(K, N, pk.rhs));
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s8(B.data(), B.size(), 123);
  fill_random_f32_pos(w_scales.data(), N, 77);
  sme::compute_ksums_s8(B.data(), K, N, w_scales.data(), w_ksums.data());
  sme::pack_s8(A.data(), M, K, pk.lhs, lp.get());
  sme::pack_s8(B.data(), K, N, pk.rhs, rp.get());
  sme::GemmParams p{M, N, K};
  sme::QuantParams qp{0, 0.05f, w_scales.data(), w_ksums.data()};

  for (auto _ : state) {
    sme::gemm_qd8p_qc8wp_f32_4vlxvl(p, lp.get(), rp.get(), C.data(), qp);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOP/s"] = benchmark::Counter(
      2.0*M*N*K, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
}

void BM_qd8_2x2(benchmark::State& state) {
  const size_t M = state.range(0), N = state.range(1), K = state.range(2);
  auto pk4 = sme::gemm_qd8_qc8w_4vlxvl_packing_params();
  auto pk2 = pk4;
  pk2.lhs.tile_rows /= 2;

  std::vector<int8_t> A(M*K), B(K*N);
  std::vector<float> C_ref(M*N), C_2x2(M*N), w_scales(N), w_ksums(N);
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s8(B.data(), B.size(), 123);
  fill_random_f32_pos(w_scales.data(), N, 77);
  sme::compute_ksums_s8(B.data(), K, N, w_scales.data(), w_ksums.data());

  auto lp4 = std::make_unique<char[]>(sme::packed_size_bytes_s8(M, K, pk4.lhs));
  auto lp2 = std::make_unique<char[]>(sme::packed_size_bytes_s8(M, K, pk2.lhs));
  auto rp  = std::make_unique<char[]>(sme::packed_size_bytes_s8(K, N, pk4.rhs));
  sme::pack_s8(A.data(), M, K, pk4.lhs, lp4.get());
  sme::pack_s8(A.data(), M, K, pk2.lhs, lp2.get());
  sme::pack_s8(B.data(), K, N, pk4.rhs, rp.get());
  sme::GemmParams p{M, N, K};
  sme::QuantParams qp{0, 0.05f, w_scales.data(), w_ksums.data()};

  sme::gemm_qd8p_qc8wp_f32_4vlxvl(p, lp4.get(), rp.get(), C_ref.data(), qp);
  run_2x2_qd8(p, lp2.get(), rp.get(), C_2x2.data(), qp);
  float d = verify(C_ref.data(), C_2x2.data(), M*N, "qd8");
  if (d > 1e-5f) { state.SkipWithError("qd8 2x2 verification failed"); return; }

  for (auto _ : state) {
    run_2x2_qd8(p, lp2.get(), rp.get(), C_2x2.data(), qp);
    benchmark::DoNotOptimize(C_2x2.data());
    benchmark::ClobberMemory();
  }
  state.counters["GFLOP/s"] = benchmark::Counter(
      2.0*M*N*K, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
}

// ---------- Registration -----------------------------------------------------

#define SIZES \
  ->Args({1024, 1024, 1024})   \
  ->Args({4096, 4096, 4096})   \
  ->Args({1024, 4096, 4096})   \
  ->Args({128, 131072, 4096})  \
  ->Unit(benchmark::kMillisecond)

BENCHMARK(BM_f32_4x1)  SIZES;
BENCHMARK(BM_f32_2x2)  SIZES;
BENCHMARK(BM_f16_4x1)  SIZES;
BENCHMARK(BM_f16_2x2)  SIZES;
BENCHMARK(BM_bf16_4x1) SIZES;
BENCHMARK(BM_bf16_2x2) SIZES;
BENCHMARK(BM_qd8_4x1)  SIZES;
BENCHMARK(BM_qd8_2x2)  SIZES;

}  // namespace
