#include "gemm_f32_2vlx2vl_sme.h"

#include <algorithm>
#include <arm_sme.h>

namespace sme {

static size_t svl_f32() {
  uint64_t svl_bytes;
  asm("rdsvl %0, #1" : "=r"(svl_bytes));
  return svl_bytes / sizeof(float);
}

GemmPackingParams gemm_f32_2vlx2vl_packing_params() {
  size_t vl = svl_f32();
  return {
      .lhs = {.tile_rows = vl * 2, .tile_cols = 1,
              .transpose_inner = false, .transpose_outer = false},
      .rhs = {.tile_rows = 1, .tile_cols = vl,
              .transpose_inner = true, .transpose_outer = true},
  };
}

// F32 activations, weight, and output. SME1. 2SVL_s x 2SVL_s tiling.
void gemm_f32p_f32p_f32_2vlx2vl_kernel(
    const GemmParams& p, const void* lhs_packed, const void* rhs_packed,
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

    for (; n < p.N; n += vl) {
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

  for (; m < p.M; m += vl) {
    const float* lhs_m = lhs_base + (m / (vl * 2)) * p.K * vl * 2
                                    + (m % (vl * 2));
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

}  // namespace sme
