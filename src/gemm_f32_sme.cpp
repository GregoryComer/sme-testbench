#include "gemm_f32_sme.h"

#include <arm_sme.h>

#ifndef PF_DIST
#define PF_DIST 512
#endif
#ifndef USE_NT_RHS
#define USE_NT_RHS 0
#endif

namespace sme {

static size_t svl_f32() {
  uint64_t svl_bytes;
  asm("rdsvl %0, #1" : "=r"(svl_bytes));
  return svl_bytes / sizeof(float);
}

size_t gemm_f32_tile_m() { return svl_f32(); }
size_t gemm_f32_tile_n() { return svl_f32(); }
size_t gemm_f32_tile_k() { return 1; }

GemmPackingParams gemm_f32_packing_params() {
  size_t svl = svl_f32();
  return {
      .lhs = {.tile_rows = svl * 4, .tile_cols = 1,
              .transpose_inner = true, .transpose_outer = false},
      .rhs = {.tile_rows = 1, .tile_cols = svl,
              .transpose_inner = false, .transpose_outer = true},
  };
}

// Unrolled 4x1 micro-kernel. Tile mapping:
//   ZA0=(m0,n0), ZA1=(m1,n0), ZA2=(m2,n0), ZA3=(m3,n0).
// Each K step: 4 LHS loads + 1 RHS load = 5 loads for 4 FMOPA.
// M epilogue handles the remainder with a 1×1 predicated loop.
// Assumes N is a multiple of svcntw().
void gemm_f32p_f32p_f32_kernel(
    const GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out) __arm_streaming __arm_inout("za") {
  const auto* lhs_base = static_cast<const float*>(lhs_packed);
  const auto* rhs_base = static_cast<const float*>(rhs_packed);
  const uint64_t vl = svcntw();
  const svbool_t pg = svptrue_b32();

  // Main body: full 4×1 M-tiles.
  const uint64_t m_body = (p.M / (vl * 4)) * (vl * 4);

  for (uint64_t m = 0; m < m_body; m += vl * 4) {
    const float* lhs_m = lhs_base + (m / (vl * 4)) * p.K * vl * 4;

    for (uint64_t n = 0; n < p.N; n += vl) {
      const float* rhs_n = rhs_base + (n / vl) * p.K * vl;

      svzero_za();

      const float* lhs_data = lhs_m;
      const float* rhs_data = rhs_n;

      for (uint64_t k = 0; k < p.K; k++) {
#if PF_DIST > 0
        svprfb(pg, lhs_data + PF_DIST * vl * 4, SV_PLDL1KEEP);
        svprfb(pg, rhs_data + PF_DIST * vl, SV_PLDL1KEEP);
#endif

        svfloat32_t lhs_col0 = svld1_f32(pg, lhs_data);
        svfloat32_t lhs_col1 = svld1_f32(pg, lhs_data + vl);
        svfloat32_t lhs_col2 = svld1_f32(pg, lhs_data + vl * 2);
        svfloat32_t lhs_col3 = svld1_f32(pg, lhs_data + vl * 3);
#if USE_NT_RHS
        svfloat32_t rhs_row0 = svldnt1_f32(pg, rhs_data);
#else
        svfloat32_t rhs_row0 = svld1_f32(pg, rhs_data);
#endif

        svmopa_za32_f32_m(0, pg, pg, lhs_col0, rhs_row0);
        svmopa_za32_f32_m(1, pg, pg, lhs_col1, rhs_row0);
        svmopa_za32_f32_m(2, pg, pg, lhs_col2, rhs_row0);
        svmopa_za32_f32_m(3, pg, pg, lhs_col3, rhs_row0);

        lhs_data += vl * 4;
        rhs_data += vl;
      }

      // Store results.
      float* out0 = out + m * p.N + n;
      float* out1 = out + (m + vl) * p.N + n;
      float* out2 = out + (m + vl * 2) * p.N + n;
      float* out3 = out + (m + vl * 3) * p.N + n;

      for (uint32_t i = 0; i < vl; i++) {
        svst1_hor_za32(0, i, pg, out0 + i * p.N);
        svst1_hor_za32(1, i, pg, out1 + i * p.N);
        svst1_hor_za32(2, i, pg, out2 + i * p.N);
        svst1_hor_za32(3, i, pg, out3 + i * p.N);
      }
    }
  }

  // M epilogue: remaining rows (< 4*vl), processed vl rows at a time
  // using a 1×1 predicated micro-kernel (ZA0 only).
  // The packed LHS tile is zero-padded by the packer so reads are safe.
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

        float* out_ptr = out + m_row * p.N + n;
        for (uint32_t i = 0; i < rows; i++) {
          svst1_hor_za32(0, i, pg, out_ptr + i * p.N);
        }
      }
    }
  }
}


}  // namespace sme
