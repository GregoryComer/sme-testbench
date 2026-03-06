#include "gemm_qd8_qc8w_4vlxvl_sme.h"

#include <arm_sme.h>

namespace sme {

static size_t svl_f32() {
  uint64_t svl_bytes;
  asm("rdsvl %0, #1" : "=r"(svl_bytes));
  return svl_bytes / sizeof(float);
}

GemmPackingParams gemm_qd8_qc8w_4vlxvl_packing_params() {
  size_t vl = svl_f32();
  // SMOPA za32 u8/s8 is rank-4: each instruction processes 4 K values.
  // LHS tiles: 4*vl rows × 4 cols (4 subtiles of vl rows, each holding
  //   4 uint8 values per row in packed order).
  // RHS tiles: 4 rows × vl cols, transposed for contiguous int8 vector loads.
  return {
      .lhs = {.tile_rows = vl * 4, .tile_cols = 4,
              .transpose_inner = false, .transpose_outer = false},
      .rhs = {.tile_rows = 4, .tile_cols = vl,
              .transpose_inner = true, .transpose_outer = true},
  };
}

void gemm_qd8p_qc8wp_f32_4vlxvl_kernel(
    const GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out, const QuantParams& qp) __arm_streaming __arm_inout("za") {

  size_t m = 0;
  auto lhs = static_cast<const int8_t*>(lhs_packed);
  auto p_K_padded = (p.K + 3) & ~0x3; // Round up to multiple of 4 (k-tile)
  svbool_t pg = svptrue_b8();

  for (; m + (svcntw() * 4) <= p.M; m += svcntw() * 4) {
    auto rhs_data = static_cast<const int8_t*>(rhs_packed);

    for (size_t n = 0; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;
      svzero_za();

      for (auto k = 0u; k < p.K; k += 4) {
        // Load 4x SVL_w x 4 blocks from LHS and 4 x SVL_w block from RHS.
        // It's prepacked to be contiguous. We can then directly pass into
        // SMOPA.
        auto lhs_block0 = svld1_s8(pg, lhs_data);
        auto lhs_block1 = svld1_s8(pg, lhs_data + svcntb());
        auto lhs_block2 = svld1_s8(pg, lhs_data + svcntb() * 2);
        auto lhs_block3 = svld1_s8(pg, lhs_data + svcntb() * 3);

        auto rhs_block0 = svld1_s8(pg, rhs_data);

        svmopa_za32_s8_m(0, pg, pg, lhs_block0, rhs_block0);
        svmopa_za32_s8_m(1, pg, pg, lhs_block1, rhs_block0);
        svmopa_za32_s8_m(2, pg, pg, lhs_block2, rhs_block0);
        svmopa_za32_s8_m(3, pg, pg, lhs_block3, rhs_block0);

        lhs_data += 4 * svcntb();
        rhs_data += svcntb();
      }

      // Extract row by row, apply act_zp and w_scales, then store.

      // Compute w_scale * a_scale.
      auto a_scale = svdup_n_f32(qp.a_scale);
      auto w_scales0 = svld1_f32(pg, &qp.w_scales[n]);
      auto scales0 = svmul_f32_x(pg, w_scales0, a_scale);

      // Compute act_zp * act_scale * ksum.
      auto ksums0 = svld1_f32(pg, &qp.w_ksums[n]);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg, svmul_f32_x(pg, ksums0, zp_f32), a_scale);

      float* out_ptr = out + m * p.N + n;

      svbool_t n_pred = svwhilelt_b32(n, (uint64_t)p.N);
      for (auto i = 0; i < svcntw(); i++) {
        // out = acc * act_scale * w_scale - act_zp * act_scale * ksum
        svint32_t row0 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i);
        svint32_t row1 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 1, i);
        svint32_t row2 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 2, i);
        svint32_t row3 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 3, i);

        svfloat32_t row0_f32 = svcvt_f32_s32_x(pg, row0);
        svfloat32_t row1_f32 = svcvt_f32_s32_x(pg, row1);
        svfloat32_t row2_f32 = svcvt_f32_s32_x(pg, row2);
        svfloat32_t row3_f32 = svcvt_f32_s32_x(pg, row3);

        // Multiply by scales then subtract zp/ksum term.
        svfloat32_t row0_w_scaled = svmul_f32_x(pg, row0_f32, scales0);
        svfloat32_t row1_w_scaled = svmul_f32_x(pg, row1_f32, scales0);
        svfloat32_t row2_w_scaled = svmul_f32_x(pg, row2_f32, scales0);
        svfloat32_t row3_w_scaled = svmul_f32_x(pg, row3_f32, scales0);

        svfloat32_t out_row0 = svsub_f32_x(pg, row0_w_scaled, scaled_ksums0);
        svfloat32_t out_row1 = svsub_f32_x(pg, row1_w_scaled, scaled_ksums0);
        svfloat32_t out_row2 = svsub_f32_x(pg, row2_w_scaled, scaled_ksums0);
        svfloat32_t out_row3 = svsub_f32_x(pg, row3_w_scaled, scaled_ksums0);

        svst1_f32(n_pred, out_ptr + i * p.N + 0 * svcntw() * p.N, out_row0);
        svst1_f32(n_pred, out_ptr + i * p.N + 1 * svcntw() * p.N, out_row1);
        svst1_f32(n_pred, out_ptr + i * p.N + 2 * svcntw() * p.N, out_row2);
        svst1_f32(n_pred, out_ptr + i * p.N + 3 * svcntw() * p.N, out_row3);
      }
    }

    lhs += svcntb() * p_K_padded; // Increment by one tile row
  }

  for (; m < p.M; m += svcntw()) {
    auto rhs_data = static_cast<const int8_t*>(rhs_packed);

    for (auto n = 0u; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;
      svzero_za();

      for (auto k = 0u; k < p.K; k += 4) {
        // Load a SVL_w x 4 block from LHS and 4 x SVL_w block from RHS.
        // It's prepacked to be contiguous. We can then directly pass into
        // SMOPA.
        auto lhs_block0 = svld1_s8(pg, lhs_data);
        auto rhs_block0 = svld1_s8(pg, rhs_data);

        svmopa_za32_s8_m(0, pg, pg, lhs_block0, rhs_block0);

        lhs_data += 4 * svcntb();  // Skip full LHS tile (4 M-subtiles × svcntb)
        rhs_data += svcntb();
      }

      // Extract row by row, apply act_zp and w_scales, then store.

      // Compute w_scale * a_scale.
      auto a_scale = svdup_n_f32(qp.a_scale);
      auto w_scales0 = svld1_f32(pg, &qp.w_scales[n]);
      auto scales0 = svmul_f32_x(pg, w_scales0, a_scale);

      // Compute act_zp * act_scale * ksum.
      auto ksums0 = svld1_f32(pg, &qp.w_ksums[n]);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg, svmul_f32_x(pg, ksums0, zp_f32), a_scale);

      float* out_ptr = out + m * p.N + n;

      svbool_t n_pred = svwhilelt_b32((uint32_t)n, (uint32_t)p.N);
      for (auto i = 0; i < svcntw(); i++) {
        // out = acc * act_scale * w_scale - act_zp * act_scale * ksum
        svint32_t row = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i);
        svfloat32_t row_f32 = svcvt_f32_s32_x(pg, row);

        // Multiply by scales then subtract zp/ksum term.
        svfloat32_t row_w_scaled = svmul_f32_x(pg, row_f32, scales0);
        svfloat32_t out_row = svsub_f32_x(pg, row_w_scaled, scaled_ksums0);

        svst1_f32(n_pred, out_ptr + i * p.N, out_row);
      }
    }

    lhs += svcntb();
  }
}

}  // namespace sme
