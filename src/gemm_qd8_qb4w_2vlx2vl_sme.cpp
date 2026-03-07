#include "gemm_qd8_qb4w_2vlx2vl_sme.h"

#include <algorithm>
#include <arm_sme.h>

namespace sme {

static size_t svl_f32() {
  uint64_t svl_bytes;
  asm("rdsvl %0, #1" : "=r"(svl_bytes));
  return svl_bytes / sizeof(float);
}

GemmPackingParams gemm_qd8_qb4w_2vlx2vl_packing_params() {
  size_t vl = svl_f32();
  return {
      .lhs = {.tile_rows = vl * 2, .tile_cols = 4,
              .transpose_inner = false, .transpose_outer = false},
      .rhs = {.tile_rows = 4, .tile_cols = vl,
              .transpose_inner = true, .transpose_outer = true},
  };
}

void gemm_qd8p_qb4wp_f32_2vlx2vl_kernel(
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
  size_t m = 0;

  // --- Full 2x2 body (2 M-subtiles × 2 N-subtiles) --------------------------
  for (; m + svcntw() * 2 <= p.M; m += svcntw() * 2) {
    size_t n = 0;
    auto rhs_col = rhs_base;
    const float* scale_col = qp.w_scales;

    // -- 2x2 N loop --
    for (; n + svcntw() * 2 <= p.N; n += svcntw() * 2) {
      auto lhs_data = lhs;
      auto r0 = rhs_col;
      auto r1 = rhs_col + rhs_n_stride;
      auto ws0_ptr = scale_col;
      auto ws1_ptr = scale_col + blocks * svcntw();

      float* op = out + m * p.N + n;

      for (size_t block = 0; block < blocks; block++) {
        svzero_za();

        for (size_t kt = 0; kt < tiles_per_block; kt++) {
          auto packed0 = svld1_s8(pg, r0);
          auto rhs0_lo = svasr_n_s8_x(pg, svlsl_n_s8_x(pg, packed0, 4), 4);
          auto rhs0_hi = svasr_n_s8_x(pg, packed0, 4);

          auto packed1 = svld1_s8(pg, r1);
          auto rhs1_lo = svasr_n_s8_x(pg, svlsl_n_s8_x(pg, packed1, 4), 4);
          auto rhs1_hi = svasr_n_s8_x(pg, packed1, 4);

          auto l0 = svld1_s8(pg, lhs_data);
          auto l1 = svld1_s8(pg, lhs_data + svcntb());
          svmopa_za32_s8_m(0, pg, pg, l0, rhs0_lo);
          svmopa_za32_s8_m(1, pg, pg, l1, rhs0_lo);
          svmopa_za32_s8_m(2, pg, pg, l0, rhs1_lo);
          svmopa_za32_s8_m(3, pg, pg, l1, rhs1_lo);
          lhs_data += 2 * svcntb();

          l0 = svld1_s8(pg, lhs_data);
          l1 = svld1_s8(pg, lhs_data + svcntb());
          svmopa_za32_s8_m(0, pg, pg, l0, rhs0_hi);
          svmopa_za32_s8_m(1, pg, pg, l1, rhs0_hi);
          svmopa_za32_s8_m(2, pg, pg, l0, rhs1_hi);
          svmopa_za32_s8_m(3, pg, pg, l1, rhs1_hi);
          lhs_data += 2 * svcntb();

          r0 += svcntb();
          r1 += svcntb();
        }

        auto ws0 = svld1_f32(pg, ws0_ptr);
        auto ws1 = svld1_f32(pg, ws1_ptr);
        ws0_ptr += svcntw();
        ws1_ptr += svcntw();

        for (uint32_t i = 0; i < svcntw(); i++) {
          auto z0 = svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i));
          auto z1 = svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 1, i));
          auto z2 = svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 2, i));
          auto z3 = svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 3, i));

          if (block == 0) {
            svst1_f32(pg, op + i * p.N, svmul_f32_x(pg, z0, ws0));
            svst1_f32(pg, op + (i + svcntw()) * p.N, svmul_f32_x(pg, z1, ws0));
            svst1_f32(pg, op + i * p.N + svcntw(), svmul_f32_x(pg, z2, ws1));
            svst1_f32(pg, op + (i + svcntw()) * p.N + svcntw(), svmul_f32_x(pg, z3, ws1));
          } else {
            svst1_f32(pg, op + i * p.N,
                svmla_f32_x(pg, svld1_f32(pg, op + i * p.N), z0, ws0));
            svst1_f32(pg, op + (i + svcntw()) * p.N,
                svmla_f32_x(pg, svld1_f32(pg, op + (i + svcntw()) * p.N), z1, ws0));
            svst1_f32(pg, op + i * p.N + svcntw(),
                svmla_f32_x(pg, svld1_f32(pg, op + i * p.N + svcntw()), z2, ws1));
            svst1_f32(pg, op + (i + svcntw()) * p.N + svcntw(),
                svmla_f32_x(pg, svld1_f32(pg, op + (i + svcntw()) * p.N + svcntw()), z3, ws1));
          }
        }
      }

      // Epilogue: subtract scaled ksums, multiply a_scale.
      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg, svld1_f32(pg, &qp.w_ksums[n]), zp_f32);
      auto scaled_ksums1 = svmul_f32_x(pg, svld1_f32(pg, &qp.w_ksums[n + svcntw()]), zp_f32);

      for (uint32_t i = 0; i < svcntw(); i++) {
        auto v0 = svld1_f32(pg, op + i * p.N);
        auto v1 = svld1_f32(pg, op + (i + svcntw()) * p.N);
        auto v2 = svld1_f32(pg, op + i * p.N + svcntw());
        auto v3 = svld1_f32(pg, op + (i + svcntw()) * p.N + svcntw());

        svst1_f32(pg, op + i * p.N,
            svmul_f32_x(pg, svsub_f32_x(pg, v0, scaled_ksums0), a_scale));
        svst1_f32(pg, op + (i + svcntw()) * p.N,
            svmul_f32_x(pg, svsub_f32_x(pg, v1, scaled_ksums0), a_scale));
        svst1_f32(pg, op + i * p.N + svcntw(),
            svmul_f32_x(pg, svsub_f32_x(pg, v2, scaled_ksums1), a_scale));
        svst1_f32(pg, op + (i + svcntw()) * p.N + svcntw(),
            svmul_f32_x(pg, svsub_f32_x(pg, v3, scaled_ksums1), a_scale));
      }

      rhs_col += 2 * rhs_n_stride;
      scale_col += 2 * blocks * svcntw();
    }

    // -- 2x1 N-tail (remaining columns after 2x2 body) --
    for (; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;
      auto r0 = rhs_col;
      auto ws0_ptr = scale_col;

      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      float* op = out + m * p.N + n;

      for (size_t block = 0; block < blocks; block++) {
        svzero_za();

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

        auto ws0 = svld1_f32(pg, ws0_ptr);
        ws0_ptr += svcntw();

        for (uint32_t i = 0; i < svcntw(); i++) {
          auto z0 = svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i));
          auto z1 = svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 1, i));

          if (block == 0) {
            svst1_f32(np, op + i * p.N, svmul_f32_x(pg, z0, ws0));
            svst1_f32(np, op + (i + svcntw()) * p.N, svmul_f32_x(pg, z1, ws0));
          } else {
            svst1_f32(np, op + i * p.N,
                svmla_f32_x(pg, svld1_f32(np, op + i * p.N), z0, ws0));
            svst1_f32(np, op + (i + svcntw()) * p.N,
                svmla_f32_x(pg, svld1_f32(np, op + (i + svcntw()) * p.N), z1, ws0));
          }
        }
      }

      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg, svld1_f32(pg, &qp.w_ksums[n]), zp_f32);

      for (uint32_t i = 0; i < svcntw(); i++) {
        auto v0 = svld1_f32(np, op + i * p.N);
        auto v1 = svld1_f32(np, op + (i + svcntw()) * p.N);
        svst1_f32(np, op + i * p.N,
            svmul_f32_x(pg, svsub_f32_x(pg, v0, scaled_ksums0), a_scale));
        svst1_f32(np, op + (i + svcntw()) * p.N,
            svmul_f32_x(pg, svsub_f32_x(pg, v1, scaled_ksums0), a_scale));
      }

      rhs_col += rhs_n_stride;
      scale_col += blocks * svcntw();
    }

    lhs += 2 * svcntb() * (K_pad / 4);
  }

  // --- M-tail 1x1 (remaining rows) ------------------------------------------
  for (; m < p.M; m += svcntw()) {
    auto rhs_col = rhs_base;
    const float* scale_col = qp.w_scales;
    size_t m_rows = std::min((size_t)svcntw(), p.M - m);

    for (size_t n = 0; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;
      auto r0 = rhs_col;
      auto ws0_ptr = scale_col;

      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      float* op = out + m * p.N + n;

      for (size_t block = 0; block < blocks; block++) {
        svzero_za();

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
          auto z0 = svcvt_f32_s32_x(pg, svread_hor_za32_s32_m(svdup_n_s32(0), pg, 0, i));

          if (block == 0) {
            svst1_f32(np, op + i * p.N, svmul_f32_x(pg, z0, ws0));
          } else {
            svst1_f32(np, op + i * p.N,
                svmla_f32_x(pg, svld1_f32(np, op + i * p.N), z0, ws0));
          }
        }
      }

      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg, svld1_f32(pg, &qp.w_ksums[n]), zp_f32);

      for (size_t i = 0; i < m_rows; i++) {
        auto v0 = svld1_f32(np, op + i * p.N);
        svst1_f32(np, op + i * p.N,
            svmul_f32_x(pg, svsub_f32_x(pg, v0, scaled_ksums0), a_scale));
      }

      rhs_col += rhs_n_stride;
      scale_col += blocks * svcntw();
    }

    lhs += svcntb();
  }
}

}  // namespace sme
