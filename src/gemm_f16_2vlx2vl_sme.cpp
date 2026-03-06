#include "gemm_f16_2vlx2vl_sme.h"

#include <algorithm>
#include <arm_sme.h>

namespace sme {

static size_t svl_f32() {
  uint64_t svl_bytes;
  asm("rdsvl %0, #1" : "=r"(svl_bytes));
  return svl_bytes / sizeof(float);
}

GemmPackingParams gemm_f16_2vlx2vl_packing_params() {
  size_t vl = svl_f32();
  // FMOPA za32 f16 is rank-2: each instruction processes 2 K values.
  // LHS tiles: 2*vl rows x 2 cols (2 subtiles of vl rows, each holding
  //   2 f16 values per row in packed order).
  // RHS tiles: 2 rows x vl cols, transposed for contiguous f16 vector loads.
  return {
      .lhs = {.tile_rows = vl * 2, .tile_cols = 2,
              .transpose_inner = false, .transpose_outer = false},
      .rhs = {.tile_rows = 2, .tile_cols = vl,
              .transpose_inner = true, .transpose_outer = true},
  };
}

// 2x2 micro-kernel for f16 GEMM with f32 accumulation.
// Tile mapping: ZA0=(m0,n0), ZA1=(m1,n0), ZA2=(m0,n1), ZA3=(m1,n1).
// Each FMOPA processes 2 K values (rank-2 widening f16->f32).
// M epilogue handles remainder with a 1x1 loop (ZA0 only).
void gemm_f16p_f16p_f16_2vlx2vl_kernel(
    const GemmParams& p, const void* lhs_packed, const void* rhs_packed,
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
    for (; n < p.N; n += vl32) {
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
      rhs_col += rhs_n_stride;
    }
  }

  // M epilogue: 1x1
  for (; m < p.M; m += vl32) {
    const float16_t* lhs_m = lhs_base + (m / (vl32 * 2)) * K_pad * vl32 * 2
                                         + (m % (vl32 * 2)) * 2;
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

}  // namespace sme
