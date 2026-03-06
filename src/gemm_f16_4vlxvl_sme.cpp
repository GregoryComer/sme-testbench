#include "gemm_f16_4vlxvl_sme.h"

#include <arm_sme.h>

#ifndef PF_DIST
#define PF_DIST 0
#endif
#ifndef PF_MODE
#define PF_MODE SV_PLDL1KEEP
#endif

namespace sme {

static size_t svl_f32() {
  uint64_t svl_bytes;
  asm("rdsvl %0, #1" : "=r"(svl_bytes));
  return svl_bytes / sizeof(float);
}

GemmPackingParams gemm_f16_4vlxvl_packing_params() {
  size_t vl = svl_f32();
  // FMOPA za32 f16 is rank-2: each instruction processes 2 K values.
  // LHS tiles: 4*vl rows × 2 cols (4 subtiles of vl rows, each holding
  //   2 f16 values per row in packed order).
  // RHS tiles: 2 rows × vl cols, transposed for contiguous f16 vector loads.
  return {
      .lhs = {.tile_rows = vl * 4, .tile_cols = 2,
              .transpose_inner = false, .transpose_outer = false},
      .rhs = {.tile_rows = 2, .tile_cols = vl,
              .transpose_inner = true, .transpose_outer = true},
  };
}

// 4x1 micro-kernel for f16 GEMM with f32 accumulation.
// Tile mapping: ZA0..ZA3 = 4 M-sub-tiles x 1 N-tile.
// Each FMOPA processes 2 K values (rank-2 widening f16→f32).
// M epilogue handles remainder with a 1x1 loop (ZA0 only).
void gemm_f16p_f16p_f16_4vlxvl_kernel(
    const GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    _Float16* out) __arm_streaming __arm_inout("za") {
  const auto lhs_base = static_cast<const float16_t*>(lhs_packed);
  const auto rhs_base = static_cast<const float16_t*>(rhs_packed);

  const uint64_t vl = svcnth();
  const uint64_t vl_f32 = svcntw();

  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg32 = svptrue_b32();

  const uint64_t K_padded = (p.K + 1) & ~1ULL;  // round up to K-tile (rank-2)
  const uint64_t m_body = (p.M / (vl_f32 * 4)) * (vl_f32 * 4);

  for (uint64_t m = 0; m < m_body; m += vl_f32 * 4) {
    const float16_t* lhs_m = lhs_base + (m / (vl_f32 * 4)) * K_padded * vl_f32 * 4;

    for (uint64_t n = 0; n < p.N; n += vl_f32) {
      const float16_t* rhs_n = rhs_base + (n / vl_f32) * K_padded * vl_f32;

      svzero_za();

      const float16_t* lhs_data = lhs_m;
      const float16_t* rhs_data = rhs_n;

      for (uint64_t k = 0; k < p.K; k += 2) {
#if PF_DIST > 0
        svprfb(pg16, lhs_data + PF_DIST * vl * 4, PF_MODE);
        svprfb(pg16, rhs_data + PF_DIST * vl, PF_MODE);
#endif
        svfloat16_t l0 = svld1_f16(pg16, lhs_data);
        svfloat16_t l1 = svld1_f16(pg16, lhs_data + vl);
        svfloat16_t l2 = svld1_f16(pg16, lhs_data + vl * 2);
        svfloat16_t l3 = svld1_f16(pg16, lhs_data + vl * 3);
        svfloat16_t r0 = svld1_f16(pg16, rhs_data);

        svmopa_za32_f16_m(0, pg16, pg16, l0, r0);
        svmopa_za32_f16_m(1, pg16, pg16, l1, r0);
        svmopa_za32_f16_m(2, pg16, pg16, l2, r0);
        svmopa_za32_f16_m(3, pg16, pg16, l3, r0);

        lhs_data += vl * 4;
        rhs_data += vl;
      }

      // Store: read ZA f32 rows → fcvt to f16 (in bottom half of each
      // 32-bit container) → ST1H truncating store (bottom 16 bits).
      const svfloat32_t f32z = svdup_n_f32(0);
      auto* dst16 = reinterpret_cast<int16_t*>(out);
      svbool_t n_pred = svwhilelt_b32(n, (uint64_t)p.N);

      for (uint32_t i = 0; i < vl_f32; i++) {
        svfloat32_t r0 = svread_hor_za32_f32_m(f32z, pg32, 0, i);
        svst1h_s32(n_pred, dst16 + (m + i) * p.N + n,
                   svreinterpret_s32_f16(svcvt_f16_f32_x(pg32, r0)));

        svfloat32_t r1 = svread_hor_za32_f32_m(f32z, pg32, 1, i);
        svst1h_s32(n_pred, dst16 + (m + vl_f32 + i) * p.N + n,
                   svreinterpret_s32_f16(svcvt_f16_f32_x(pg32, r1)));

        svfloat32_t r2 = svread_hor_za32_f32_m(f32z, pg32, 2, i);
        svst1h_s32(n_pred, dst16 + (m + vl_f32 * 2 + i) * p.N + n,
                   svreinterpret_s32_f16(svcvt_f16_f32_x(pg32, r2)));

        svfloat32_t r3 = svread_hor_za32_f32_m(f32z, pg32, 3, i);
        svst1h_s32(n_pred, dst16 + (m + vl_f32 * 3 + i) * p.N + n,
                   svreinterpret_s32_f16(svcvt_f16_f32_x(pg32, r3)));
      }
    }
  }

  // M epilogue: remaining rows (< 4*vl), processed vl rows at a time
  // using a 1x1 predicated micro-kernel (ZA0 only).
  if (m_body < p.M) {
    const float16_t* lhs_m =
        lhs_base + (m_body / (vl_f32 * 4)) * K_padded * vl_f32 * 4;

    for (uint64_t s = 0; m_body + s * vl_f32 < p.M; s++) {
      const uint64_t m_row = m_body + s * vl_f32;
      const svbool_t m_pred = svwhilelt_b32(m_row, (uint64_t)p.M);
      const uint64_t rem = p.M - m_row;
      const uint64_t rows = rem < vl_f32 ? rem : vl_f32;

      for (uint64_t n = 0; n < p.N; n += vl_f32) {
        const float16_t* rhs_n = rhs_base + (n / vl_f32) * K_padded * vl_f32;

        svzero_za();

        const float16_t* lhs_data = lhs_m + s * vl;
        const float16_t* rhs_data = rhs_n;

        for (uint64_t k = 0; k < p.K; k += 2) {
          svfloat16_t lhs_col = svld1_f16(pg16, lhs_data);
          svfloat16_t rhs_row = svld1_f16(pg16, rhs_data);

          svmopa_za32_f16_m(0, /*m_pred*/pg16, pg16, lhs_col, rhs_row);

          lhs_data += vl * 4;
          rhs_data += vl;
        }

        const svfloat32_t ez32 = svdup_n_f32(0);
        auto* edst16 = reinterpret_cast<int16_t*>(out);
        svbool_t n_pred = svwhilelt_b32(n, (uint64_t)p.N);
        for (uint32_t i = 0; i < rows; i++) {
          svfloat32_t r = svread_hor_za32_f32_m(ez32, pg32, 0, i);
          svst1h_s32(n_pred, edst16 + (m_row + i) * p.N + n,
                     svreinterpret_s32_f16(svcvt_f16_f32_x(pg32, r)));
        }
      }
    }
  }
}

}  // namespace sme
