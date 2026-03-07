#include "gemm_qd8_qb4w_4vlxvl_sme.h"

#include <algorithm>
#include <arm_sme.h>
#include <cstring>

namespace sme {

static size_t svl_f32() {
  uint64_t svl_bytes;
  asm("rdsvl %0, #1" : "=r"(svl_bytes));
  return svl_bytes / sizeof(float);
}

GemmPackingParams gemm_qd8_qb4w_4vlxvl_packing_params() {
  size_t vl = svl_f32();
  // Identical to qc4w: int8 LHS with rank-4 SMOPA layout,
  // nibble-packed RHS with transposed tile order.
  return {
      .lhs = {.tile_rows = vl * 4, .tile_cols = 4,
              .transpose_inner = false, .transpose_outer = false},
      .rhs = {.tile_rows = 4, .tile_cols = vl,
              .transpose_inner = true, .transpose_outer = true},
  };
}

void gemm_qd8p_qb4wp_f32_4vlxvl_kernel(
    const GemmParams& p, 
    const void* lhs_packed, 
    const void* rhs_packed,
    float* out, 
    const BlockQuantParams& qp, 
    float* scratch // SVL_h x SVL_h * 4 element buffer
) __arm_streaming __arm_inout("za") {
  size_t m = 0;
  auto lhs = static_cast<const int8_t*>(lhs_packed);
  auto p_K_padded = (p.K + 3) & ~0x3;  // Round up to k-tile size (4)
  auto blocks = p.K / qp.group_size;
  auto tiles_per_block = qp.group_size / 8; // K_TILE=8, we assume group size is a multiple of 8 (TODO Check this?)

  svbool_t pg = svptrue_b8();

  // --- Full M-tile loop (4 subtiles) ----------------------------------------
  for (; m + (svcntw() * 4) <= p.M; m += svcntw() * 4) {
    auto rhs_data = static_cast<const int8_t*>(rhs_packed);
    auto scale_data = qp.w_scales;

    for (auto n = 0u; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;

      // Zero float accumulator.
      std::fill(scratch, scratch + svcntw() * svcntw() * 4, 0);

      size_t k = 0;
      while (k < p.K) {
        // Zero integer accumulator.
        svzero_za();

        for (auto k_tile = 0; k_tile < tiles_per_block; k_tile++) {
          // Load nibble-packed RHS and unpack to two s8 vectors.
          auto packed = svld1_s8(pg, rhs_data);
          auto rhs_lo = svasr_n_s8_x(pg, svlsl_n_s8_x(pg, packed, 4), 4);
          auto rhs_hi = svasr_n_s8_x(pg, packed, 4);

          // K-tile 0 (lower nibble)
          auto lhs0 = svld1_s8(pg, lhs_data);
          auto lhs1 = svld1_s8(pg, lhs_data + svcntb());
          auto lhs2 = svld1_s8(pg, lhs_data + svcntb() * 2);
          auto lhs3 = svld1_s8(pg, lhs_data + svcntb() * 3);

          svmopa_za32_s8_m(0, pg, pg, lhs0, rhs_lo);
          svmopa_za32_s8_m(1, pg, pg, lhs1, rhs_lo);
          svmopa_za32_s8_m(2, pg, pg, lhs2, rhs_lo);
          svmopa_za32_s8_m(3, pg, pg, lhs3, rhs_lo);
          lhs_data += 4 * svcntb();

          // K-tile 1 (upper nibble)
          lhs0 = svld1_s8(pg, lhs_data);
          lhs1 = svld1_s8(pg, lhs_data + svcntb());
          lhs2 = svld1_s8(pg, lhs_data + svcntb() * 2);
          lhs3 = svld1_s8(pg, lhs_data + svcntb() * 3);

          svmopa_za32_s8_m(0, pg, pg, lhs0, rhs_hi);
          svmopa_za32_s8_m(1, pg, pg, lhs1, rhs_hi);
          svmopa_za32_s8_m(2, pg, pg, lhs2, rhs_hi);
          svmopa_za32_s8_m(3, pg, pg, lhs3, rhs_hi);
          lhs_data += 4 * svcntb();

          rhs_data += svcntb();

          k += 8;
        }

        // Apply per-block scales and accumulate into f32.
        auto scale_data0 = svld1_f32(pg, scale_data);
        scale_data += svcntw();

        auto acc_stride = svcntw() * svcntw();
        auto acc0 = scratch;
        auto acc1 = scratch + acc_stride;
        auto acc2 = scratch + acc_stride * 2;
        auto acc3 = scratch + acc_stride * 3;

        for (auto i = 0; i < svcntw(); i++) {
          svint32_t row0 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i);
          svint32_t row1 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 1, i);
          svint32_t row2 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 2, i);
          svint32_t row3 = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 3, i);

          svfloat32_t row0_f32 = svcvt_f32_s32_x(pg, row0);
          svfloat32_t row1_f32 = svcvt_f32_s32_x(pg, row1);
          svfloat32_t row2_f32 = svcvt_f32_s32_x(pg, row2);
          svfloat32_t row3_f32 = svcvt_f32_s32_x(pg, row3);

          svfloat32_t row_w_scaled0 = svmul_f32_x(pg, row0_f32, scale_data0);
          svfloat32_t row_w_scaled1 = svmul_f32_x(pg, row1_f32, scale_data0);
          svfloat32_t row_w_scaled2 = svmul_f32_x(pg, row2_f32, scale_data0);
          svfloat32_t row_w_scaled3 = svmul_f32_x(pg, row3_f32, scale_data0);

          svfloat32_t acc_row0 = svld1_f32(pg, acc0);
          svfloat32_t acc_row1 = svld1_f32(pg, acc1);
          svfloat32_t acc_row2 = svld1_f32(pg, acc2);
          svfloat32_t acc_row3 = svld1_f32(pg, acc3);

          acc_row0 = svadd_f32_m(pg, acc_row0, row_w_scaled0);
          acc_row1 = svadd_f32_m(pg, acc_row1, row_w_scaled1);
          acc_row2 = svadd_f32_m(pg, acc_row2, row_w_scaled2);
          acc_row3 = svadd_f32_m(pg, acc_row3, row_w_scaled3);

          svst1_f32(pg, acc0, acc_row0);
          svst1_f32(pg, acc1, acc_row1);
          svst1_f32(pg, acc2, acc_row2);
          svst1_f32(pg, acc3, acc_row3);

          acc0 += svcntw();
          acc1 += svcntw();
          acc2 += svcntw();
          acc3 += svcntw();
        }
      }
       
      // --- Epilogue ---
      auto a_scale = svdup_n_f32(qp.a_scale);
      auto ksums0 = svld1_f32(pg, &qp.w_ksums[n]);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg, ksums0, zp_f32);

      float* out_ptr = out + m * p.N + n;

      svbool_t n_pred = svwhilelt_b32((uint32_t)n, (uint32_t)p.N);
      auto acc_stride = svcntw() * svcntw();
      auto acc0 = scratch;
      auto acc1 = scratch + acc_stride;
      auto acc2 = scratch + acc_stride * 2;
      auto acc3 = scratch + acc_stride * 3;
      for (auto i = 0; i < svcntw(); i++) {
        svfloat32_t row0 = svld1_f32(pg, acc0);
        svfloat32_t row1 = svld1_f32(pg, acc1);
        svfloat32_t row2 = svld1_f32(pg, acc2);
        svfloat32_t row3 = svld1_f32(pg, acc3);

        svfloat32_t out_row0 = svsub_f32_x(pg, row0, scaled_ksums0);
        svfloat32_t out_row1 = svsub_f32_x(pg, row1, scaled_ksums0);
        svfloat32_t out_row2 = svsub_f32_x(pg, row2, scaled_ksums0);
        svfloat32_t out_row3 = svsub_f32_x(pg, row3, scaled_ksums0);

        out_row0 = svmul_f32_x(pg, out_row0, a_scale);
        out_row1 = svmul_f32_x(pg, out_row1, a_scale);
        out_row2 = svmul_f32_x(pg, out_row2, a_scale);
        out_row3 = svmul_f32_x(pg, out_row3, a_scale);

        svst1_f32(n_pred, out_ptr + i * p.N, out_row0);
        svst1_f32(n_pred, out_ptr + (i + svcntw()) * p.N, out_row1);
        svst1_f32(n_pred, out_ptr + (i + svcntw() * 2) * p.N, out_row2);
        svst1_f32(n_pred, out_ptr + (i + svcntw() * 3) * p.N, out_row3);

        acc0 += svcntw();
        acc1 += svcntw();
        acc2 += svcntw();
        acc3 += svcntw();
      }
    }

    lhs += svcntb() * p_K_padded;
  }
  
  // --- Partial M-tile loop (1 subtile) --------------------------------------
  for (; m < p.M; m += svcntw()) {
    auto rhs_data = static_cast<const int8_t*>(rhs_packed);
    auto scale_data = qp.w_scales;

    for (auto n = 0u; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;

      // Zero float accumulator.
      std::fill(scratch, scratch + svcntw() * svcntw() * 4, 0);

      size_t k = 0;
      while (k < p.K) {
        // Zero integer accumulator.
        svzero_za();

        for (auto k_tile = 0; k_tile < tiles_per_block; k_tile++) {
          auto packed = svld1_s8(pg, rhs_data);
          auto rhs_lo = svasr_n_s8_x(pg, svlsl_n_s8_x(pg, packed, 4), 4);
          auto rhs_hi = svasr_n_s8_x(pg, packed, 4);

          auto lhs_block = svld1_s8(pg, lhs_data);
          svmopa_za32_s8_m(0, pg, pg, lhs_block, rhs_lo);
          lhs_data += 4 * svcntb();

          lhs_block = svld1_s8(pg, lhs_data);
          svmopa_za32_s8_m(0, pg, pg, lhs_block, rhs_hi);
          lhs_data += 4 * svcntb();

          rhs_data += svcntb();

          k += 8;
        }

        // Apply per-block scales and accumulate into f32.
        auto scale_data0 = svld1_f32(pg, scale_data);
        scale_data += svcntw();

        auto acc = scratch;
        for (auto i = 0; i < svcntw(); i++) {
          svint32_t row = svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i);
          svfloat32_t row_f32 = svcvt_f32_s32_x(pg, row);

          svfloat32_t row_w_scaled = svmul_f32_x(pg, row_f32, scale_data0);
          svfloat32_t acc_row = svld1_f32(pg, acc);
          acc_row = svadd_f32_m(pg, acc_row, row_w_scaled);
          svst1_f32(pg, acc, acc_row);
          acc += svcntw();
        }
      }
       
      // --- Epilogue ---
      auto a_scale = svdup_n_f32(qp.a_scale);
      auto ksums0 = svld1_f32(pg, &qp.w_ksums[n]);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg, ksums0, zp_f32);

      float* out_ptr = out + m * p.N + n;

      svbool_t n_pred = svwhilelt_b32((uint32_t)n, (uint32_t)p.N);
      size_t m_rows = std::min((size_t)svcntw(), p.M - m);
      auto acc = scratch;
      for (size_t i = 0; i < m_rows; i++) {
        svfloat32_t row = svld1_f32(pg, acc);
        svfloat32_t out_row = svsub_f32_x(pg, row, scaled_ksums0);
        out_row = svmul_f32_x(pg, out_row, a_scale);
        svst1_f32(n_pred, out_ptr + i * p.N, out_row);

        acc += svcntw();
      }
    }

    lhs += svcntb();
  }
}

}  // namespace sme
