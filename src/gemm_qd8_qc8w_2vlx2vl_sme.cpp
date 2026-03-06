#include "gemm_qd8_qc8w_2vlx2vl_sme.h"

#include <arm_sme.h>

namespace sme {

static size_t svl_f32() {
  uint64_t svl_bytes;
  asm("rdsvl %0, #1" : "=r"(svl_bytes));
  return svl_bytes / sizeof(float);
}

GemmPackingParams gemm_qd8_qc8w_2vlx2vl_packing_params() {
  size_t vl = svl_f32();
  // SMOPA za32 s8 is rank-4: each instruction processes 4 K values.
  // LHS tiles: 2*vl rows x 4 cols (2 subtiles of vl rows, each holding
  //   4 int8 values per row in packed order).
  // RHS tiles: 4 rows x vl cols, transposed for contiguous int8 vector loads.
  return {
      .lhs = {.tile_rows = vl * 2, .tile_cols = 4,
              .transpose_inner = false, .transpose_outer = false},
      .rhs = {.tile_rows = 4, .tile_cols = vl,
              .transpose_inner = true, .transpose_outer = true},
  };
}

// 2x2 micro-kernel for qd8xqc8w->f32 GEMM with s32 accumulation.
// Tile mapping: ZA0=(m0,n0), ZA1=(m1,n0), ZA2=(m0,n1), ZA3=(m1,n1).
// Each SMOPA processes 4 K values (rank-4 widening s8->s32).
// M epilogue handles remainder with a 1x1 loop (ZA0 only).
void gemm_qd8p_qc8wp_f32_2vlx2vl_kernel(
    const GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out, const QuantParams& qp) __arm_streaming __arm_inout("za") {

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

    for (; n < p.N; n += svcntw()) {
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
      rhs_col += rhs_n_stride;
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

}  // namespace sme
