#include "gemm_qd8_qb4w_2vlxvl_sme.h"

#include <algorithm>
#include <arm_sme.h>

namespace sme {

static size_t svl_f32() {
  uint64_t svl_bytes;
  asm("rdsvl %0, #1" : "=r"(svl_bytes));
  return svl_bytes / sizeof(float);
}

GemmPackingParams gemm_qd8_qb4w_2vlxvl_packing_params() {
  size_t vl = svl_f32();
  return {
      .lhs = {.tile_rows = vl * 2, .tile_cols = 4,
              .transpose_inner = false, .transpose_outer = false},
      .rhs = {.tile_rows = 4, .tile_cols = vl,
              .transpose_inner = true, .transpose_outer = true},
  };
}

// ZA tile allocation:
//   ZA0 (int32): SMOPA accumulator for M-subtile 0
//   ZA1 (int32): SMOPA accumulator for M-subtile 1
//   ZA2 (float32): cross-group accumulator for M-subtile 0
//   ZA3 (float32): cross-group accumulator for M-subtile 1
//
// After each group's SMOPA inner loop, the int32 results in ZA0/ZA1 are
// converted to float, scaled by the group weight scale, and accumulated
// into ZA2/ZA3.  ZA0/ZA1 are then zeroed for the next group.
//
// This eliminates all memory traffic between groups — the intermediate
// float accumulation stays entirely in ZA tiles until the final epilogue.

void gemm_qd8p_qb4wp_f32_2vlxvl_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams& qp) __arm_streaming __arm_inout("za") {

  auto lhs = static_cast<const int8_t*>(lhs_packed);
  auto rhs_base = static_cast<const int8_t*>(rhs_packed);
  const size_t K_pad = (p.K + 3) & ~size_t(3);
  const size_t rhs_n_stride = (K_pad / 8) * svcntb();
  const size_t blocks = p.K / qp.group_size;
  const size_t tiles_per_block = qp.group_size / 8;

  svbool_t pg = svptrue_b8();
  auto zero_s32 = svdup_n_s32(0);
  auto zero_f32 = svdup_n_f32(0.0f);
  size_t m = 0;

  // --- Full 2x1 body (2 M-subtiles × 1 N-subtile) ----------------------------
  for (; m + svcntw() * 2 <= p.M; m += svcntw() * 2) {
    auto rhs_col = rhs_base;
    const float* scale_col = qp.w_scales;

    for (size_t n = 0; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;
      auto r0 = rhs_col;
      auto ws0_ptr = scale_col;
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);

      svzero_za();  // Zero all 4 tiles at start of (m,n) block.

      for (size_t block = 0; block < blocks; block++) {
        // SMOPA inner loop: accumulate into ZA0/ZA1 (int32).
        for (size_t kt = 0; kt < tiles_per_block; kt++) {
          auto packed0 = svld1_s8(pg, r0);
          auto rhs0_lo = svasr_n_s8_x(pg, svlsl_n_s8_x(pg, packed0, 4), 4);
          auto rhs0_hi = svasr_n_s8_x(pg, packed0, 4);

          auto l0 = svld1_s8(pg, lhs_data);
          auto l1 = svld1_s8(pg, lhs_data + svcntb());
          svmopa_za32_s8_m(0, pg, pg, l0, rhs0_lo);
          svmopa_za32_s8_m(1, pg, pg, l1, rhs0_lo);
          lhs_data += 2 * svcntb();

          l0 = svld1_s8(pg, lhs_data);
          l1 = svld1_s8(pg, lhs_data + svcntb());
          svmopa_za32_s8_m(0, pg, pg, l0, rhs0_hi);
          svmopa_za32_s8_m(1, pg, pg, l1, rhs0_hi);
          lhs_data += 2 * svcntb();

          r0 += svcntb();
        }

        // Accumulate ZA0/ZA1 (int32) → ZA2/ZA3 (float32), then zero ZA0/ZA1.
        auto ws0 = svld1_f32(pg, ws0_ptr);
        ws0_ptr += svcntw();

        for (uint32_t i = 0; i < svcntw(); i++) {
          auto z0 = svcvt_f32_s32_x(pg,
              svread_hor_za32_s32_m(zero_s32, pg, 0, i));
          auto acc0 = svread_hor_za32_f32_m(zero_f32, pg, 2, i);
          svwrite_hor_za32_f32_m(2, i, pg, svmla_f32_x(pg, acc0, z0, ws0));
          svwrite_hor_za32_s32_m(0, i, pg, zero_s32);

          auto z1 = svcvt_f32_s32_x(pg,
              svread_hor_za32_s32_m(zero_s32, pg, 1, i));
          auto acc1 = svread_hor_za32_f32_m(zero_f32, pg, 3, i);
          svwrite_hor_za32_f32_m(3, i, pg, svmla_f32_x(pg, acc1, z1, ws0));
          svwrite_hor_za32_s32_m(1, i, pg, zero_s32);
        }
      }

      // Epilogue: read ZA2/ZA3, apply ksums + a_scale, store to output.
      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg,
          svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums = svmul_f32_x(pg, svld1_f32(pg, &qp.w_ksums[n]), zp_f32);

      float* op = out + m * p.N + n;
      for (uint32_t i = 0; i < svcntw(); i++) {
        auto v0 = svread_hor_za32_f32_m(zero_f32, pg, 2, i);
        auto v1 = svread_hor_za32_f32_m(zero_f32, pg, 3, i);
        svst1_f32(np, op + i * p.N,
            svmul_f32_x(pg, svsub_f32_x(pg, v0, scaled_ksums), a_scale));
        svst1_f32(np, op + (i + svcntw()) * p.N,
            svmul_f32_x(pg, svsub_f32_x(pg, v1, scaled_ksums), a_scale));
      }

      rhs_col += rhs_n_stride;
      scale_col += blocks * svcntw();
    }

    lhs += 2 * svcntb() * (K_pad / 4);
  }

  // --- M-tail 1x1 (remaining rows) -------------------------------------------
  for (; m < p.M; m += svcntw()) {
    auto rhs_col = rhs_base;
    const float* scale_col = qp.w_scales;
    size_t m_rows = std::min((size_t)svcntw(), p.M - m);

    for (size_t n = 0; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;
      auto r0 = rhs_col;
      auto ws0_ptr = scale_col;
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);

      svzero_za();

      for (size_t block = 0; block < blocks; block++) {
        for (size_t kt = 0; kt < tiles_per_block; kt++) {
          auto packed0 = svld1_s8(pg, r0);
          auto rhs0_lo = svasr_n_s8_x(pg, svlsl_n_s8_x(pg, packed0, 4), 4);
          auto rhs0_hi = svasr_n_s8_x(pg, packed0, 4);

          auto l0 = svld1_s8(pg, lhs_data);
          svmopa_za32_s8_m(0, pg, pg, l0, rhs0_lo);
          lhs_data += 2 * svcntb();

          l0 = svld1_s8(pg, lhs_data);
          svmopa_za32_s8_m(0, pg, pg, l0, rhs0_hi);
          lhs_data += 2 * svcntb();

          r0 += svcntb();
        }

        auto ws0 = svld1_f32(pg, ws0_ptr);
        ws0_ptr += svcntw();

        for (size_t i = 0; i < m_rows; i++) {
          auto z0 = svcvt_f32_s32_x(pg,
              svread_hor_za32_s32_m(zero_s32, pg, 0, i));
          auto acc0 = svread_hor_za32_f32_m(zero_f32, pg, 2, i);
          svwrite_hor_za32_f32_m(2, i, pg, svmla_f32_x(pg, acc0, z0, ws0));
          svwrite_hor_za32_s32_m(0, i, pg, zero_s32);
        }
      }

      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg,
          svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums = svmul_f32_x(pg, svld1_f32(pg, &qp.w_ksums[n]), zp_f32);

      float* op = out + m * p.N + n;
      for (size_t i = 0; i < m_rows; i++) {
        auto v0 = svread_hor_za32_f32_m(zero_f32, pg, 2, i);
        svst1_f32(np, op + i * p.N,
            svmul_f32_x(pg, svsub_f32_x(pg, v0, scaled_ksums), a_scale));
      }

      rhs_col += rhs_n_stride;
      scale_col += blocks * svcntw();
    }

    lhs += svcntb();
  }
}

}  // namespace sme
