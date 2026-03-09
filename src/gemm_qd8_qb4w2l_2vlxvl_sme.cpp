#include "gemm_qd8_qb4w2l_2vlxvl_sme.h"

#include <algorithm>
#include <arm_sme.h>

namespace sme {

static size_t svl_f32() {
  uint64_t svl_bytes;
  asm("rdsvl %0, #1" : "=r"(svl_bytes));
  return svl_bytes / sizeof(float);
}

GemmPackingParams gemm_qd8_qb4w2l_2vlxvl_packing_params() {
  size_t vl = svl_f32();
  return {
      .lhs = {.tile_rows = vl * 2, .tile_cols = 4,
              .transpose_inner = false, .transpose_outer = false},
      .rhs = {.tile_rows = 4, .tile_cols = vl,
              .transpose_inner = true, .transpose_outer = true},
  };
}

// INT8 activations, 2-level blockwise INT4 weight, F32 output. SME1. 2SVL_s x 2SVL_s tiling.
void gemm_qd8p_qb4w2lp_f32_2vlxvl_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams2L& qp) __arm_streaming __arm_inout("za") {

  auto lhs = static_cast<const int8_t*>(lhs_packed);
  auto rhs_base = static_cast<const int8_t*>(rhs_packed);
  const size_t K_pad = (p.K + 3) & ~size_t(3);
  const size_t rhs_n_stride = (K_pad / 8) * svcntb();
  const size_t inner_per_outer = qp.outer_group_size / qp.inner_group_size;
  const size_t num_inner = K_pad / qp.inner_group_size;
  const size_t num_outer = K_pad / qp.outer_group_size;
  const size_t tiles_per_inner = qp.inner_group_size / 8;

  svbool_t pg = svptrue_b8();
  svbool_t pg32 = svptrue_b32();
  // Predicate for loading svcntw() int8 scale values (first VL/4 byte lanes).
  svbool_t pg_n = svwhilelt_b8((uint64_t)0, svcntw());

  size_t m = 0;

  for (; m + svcntw() * 2 <= p.M; m += svcntw() * 2) {
    size_t n = 0;
    auto rhs_col = rhs_base;
    const int8_t* inner_scale_col = qp.inner_scales;
    const float* outer_scale_col = qp.outer_scales;

    for (; n + svcntw() * 2 <= p.N; n += svcntw() * 2) {
      auto lhs_data = lhs;
      auto r0 = rhs_col;
      auto r1 = rhs_col + rhs_n_stride;
      auto is0_ptr = inner_scale_col;
      auto is1_ptr = inner_scale_col + num_inner * svcntw();
      auto os0_ptr = outer_scale_col;
      auto os1_ptr = outer_scale_col + num_outer * svcntw();

      float* op = out + m * p.N + n;

      for (size_t ob = 0; ob < num_outer; ob++) {
        svzero_za();

        for (size_t ib_local = 0; ib_local < inner_per_outer; ib_local++) {
          // Load inner scale, replicate 4x for SMOPA K-lane layout.
          auto is0_raw = svld1_s8(pg_n, is0_ptr);
          auto is0_pairs = svzip1_s8(is0_raw, is0_raw);
          auto is0_quads = svzip1_s8(is0_pairs, is0_pairs);
          is0_ptr += svcntw();

          auto is1_raw = svld1_s8(pg_n, is1_ptr);
          auto is1_pairs = svzip1_s8(is1_raw, is1_raw);
          auto is1_quads = svzip1_s8(is1_pairs, is1_pairs);
          is1_ptr += svcntw();

          for (size_t kt = 0; kt < tiles_per_inner; kt++) {
            // Unpack nibbles and pre-scale by inner scale.
            auto packed0 = svld1_s8(pg, r0);
            auto rhs0_lo = svmul_s8_x(pg,
                svasr_n_s8_x(pg, svlsl_n_s8_x(pg, packed0, 4), 4), is0_quads);
            auto rhs0_hi = svmul_s8_x(pg,
                svasr_n_s8_x(pg, packed0, 4), is0_quads);

            auto packed1 = svld1_s8(pg, r1);
            auto rhs1_lo = svmul_s8_x(pg,
                svasr_n_s8_x(pg, svlsl_n_s8_x(pg, packed1, 4), 4), is1_quads);
            auto rhs1_hi = svmul_s8_x(pg,
                svasr_n_s8_x(pg, packed1, 4), is1_quads);

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
        }

        // Outer block reduction: read ZA, cvt f32, scale, write/FMA output.
        auto os0 = svld1_f32(pg32, os0_ptr);
        auto os1 = svld1_f32(pg32, os1_ptr);
        os0_ptr += svcntw();
        os1_ptr += svcntw();

        for (uint32_t i = 0; i < svcntw(); i++) {
          auto z0 = svcvt_f32_s32_x(pg32,
              svread_hor_za32_s32_m(svdup_n_s32(0), pg32, 0, i));
          auto z1 = svcvt_f32_s32_x(pg32,
              svread_hor_za32_s32_m(svdup_n_s32(0), pg32, 1, i));
          auto z2 = svcvt_f32_s32_x(pg32,
              svread_hor_za32_s32_m(svdup_n_s32(0), pg32, 2, i));
          auto z3 = svcvt_f32_s32_x(pg32,
              svread_hor_za32_s32_m(svdup_n_s32(0), pg32, 3, i));

          if (ob == 0) {
            svst1_f32(pg32, op + i * p.N, svmul_f32_x(pg32, z0, os0));
            svst1_f32(pg32, op + (i + svcntw()) * p.N, svmul_f32_x(pg32, z1, os0));
            svst1_f32(pg32, op + i * p.N + svcntw(), svmul_f32_x(pg32, z2, os1));
            svst1_f32(pg32, op + (i + svcntw()) * p.N + svcntw(), svmul_f32_x(pg32, z3, os1));
          } else {
            svst1_f32(pg32, op + i * p.N,
                svmla_f32_x(pg32, svld1_f32(pg32, op + i * p.N), z0, os0));
            svst1_f32(pg32, op + (i + svcntw()) * p.N,
                svmla_f32_x(pg32, svld1_f32(pg32, op + (i + svcntw()) * p.N), z1, os0));
            svst1_f32(pg32, op + i * p.N + svcntw(),
                svmla_f32_x(pg32, svld1_f32(pg32, op + i * p.N + svcntw()), z2, os1));
            svst1_f32(pg32, op + (i + svcntw()) * p.N + svcntw(),
                svmla_f32_x(pg32, svld1_f32(pg32, op + (i + svcntw()) * p.N + svcntw()), z3, os1));
          }
        }
      }

      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg32,
          svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg32, svld1_f32(pg32, &qp.w_ksums[n]), zp_f32);
      auto scaled_ksums1 = svmul_f32_x(pg32, svld1_f32(pg32, &qp.w_ksums[n + svcntw()]), zp_f32);

      for (uint32_t i = 0; i < svcntw(); i++) {
        auto v0 = svld1_f32(pg32, op + i * p.N);
        auto v1 = svld1_f32(pg32, op + (i + svcntw()) * p.N);
        auto v2 = svld1_f32(pg32, op + i * p.N + svcntw());
        auto v3 = svld1_f32(pg32, op + (i + svcntw()) * p.N + svcntw());

        svst1_f32(pg32, op + i * p.N,
            svmul_f32_x(pg32, svsub_f32_x(pg32, v0, scaled_ksums0), a_scale));
        svst1_f32(pg32, op + (i + svcntw()) * p.N,
            svmul_f32_x(pg32, svsub_f32_x(pg32, v1, scaled_ksums0), a_scale));
        svst1_f32(pg32, op + i * p.N + svcntw(),
            svmul_f32_x(pg32, svsub_f32_x(pg32, v2, scaled_ksums1), a_scale));
        svst1_f32(pg32, op + (i + svcntw()) * p.N + svcntw(),
            svmul_f32_x(pg32, svsub_f32_x(pg32, v3, scaled_ksums1), a_scale));
      }

      rhs_col += 2 * rhs_n_stride;
      inner_scale_col += 2 * num_inner * svcntw();
      outer_scale_col += 2 * num_outer * svcntw();
    }

    for (; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;
      auto r0 = rhs_col;
      auto is0_ptr = inner_scale_col;
      auto os0_ptr = outer_scale_col;

      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      float* op = out + m * p.N + n;

      for (size_t ob = 0; ob < num_outer; ob++) {
        svzero_za();

        for (size_t ib_local = 0; ib_local < inner_per_outer; ib_local++) {
          auto is0_raw = svld1_s8(pg_n, is0_ptr);
          auto is0_pairs = svzip1_s8(is0_raw, is0_raw);
          auto is0_quads = svzip1_s8(is0_pairs, is0_pairs);
          is0_ptr += svcntw();

          for (size_t kt = 0; kt < tiles_per_inner; kt++) {
            auto packed0 = svld1_s8(pg, r0);
            auto rhs0_lo = svmul_s8_x(pg,
                svasr_n_s8_x(pg, svlsl_n_s8_x(pg, packed0, 4), 4), is0_quads);
            auto rhs0_hi = svmul_s8_x(pg,
                svasr_n_s8_x(pg, packed0, 4), is0_quads);

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
        }

        auto os0 = svld1_f32(pg32, os0_ptr);
        os0_ptr += svcntw();

        for (uint32_t i = 0; i < svcntw(); i++) {
          auto z0 = svcvt_f32_s32_x(pg32,
              svread_hor_za32_s32_m(svdup_n_s32(0), pg32, 0, i));
          auto z1 = svcvt_f32_s32_x(pg32,
              svread_hor_za32_s32_m(svdup_n_s32(0), pg32, 1, i));

          if (ob == 0) {
            svst1_f32(np, op + i * p.N, svmul_f32_x(pg32, z0, os0));
            svst1_f32(np, op + (i + svcntw()) * p.N, svmul_f32_x(pg32, z1, os0));
          } else {
            svst1_f32(np, op + i * p.N,
                svmla_f32_x(pg32, svld1_f32(np, op + i * p.N), z0, os0));
            svst1_f32(np, op + (i + svcntw()) * p.N,
                svmla_f32_x(pg32, svld1_f32(np, op + (i + svcntw()) * p.N), z1, os0));
          }
        }
      }

      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg32,
          svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg32, svld1_f32(pg32, &qp.w_ksums[n]), zp_f32);

      for (uint32_t i = 0; i < svcntw(); i++) {
        auto v0 = svld1_f32(np, op + i * p.N);
        auto v1 = svld1_f32(np, op + (i + svcntw()) * p.N);
        svst1_f32(np, op + i * p.N,
            svmul_f32_x(pg32, svsub_f32_x(pg32, v0, scaled_ksums0), a_scale));
        svst1_f32(np, op + (i + svcntw()) * p.N,
            svmul_f32_x(pg32, svsub_f32_x(pg32, v1, scaled_ksums0), a_scale));
      }

      rhs_col += rhs_n_stride;
      inner_scale_col += num_inner * svcntw();
      outer_scale_col += num_outer * svcntw();
    }

    lhs += 2 * svcntb() * (K_pad / 4);
  }

  for (; m < p.M; m += svcntw()) {
    auto rhs_col = rhs_base;
    const int8_t* inner_scale_col = qp.inner_scales;
    const float* outer_scale_col = qp.outer_scales;
    size_t m_rows = std::min((size_t)svcntw(), p.M - m);

    for (size_t n = 0; n < p.N; n += svcntw()) {
      auto lhs_data = lhs;
      auto r0 = rhs_col;
      auto is0_ptr = inner_scale_col;
      auto os0_ptr = outer_scale_col;

      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);
      float* op = out + m * p.N + n;

      for (size_t ob = 0; ob < num_outer; ob++) {
        svzero_za();

        for (size_t ib_local = 0; ib_local < inner_per_outer; ib_local++) {
          auto is0_raw = svld1_s8(pg_n, is0_ptr);
          auto is0_pairs = svzip1_s8(is0_raw, is0_raw);
          auto is0_quads = svzip1_s8(is0_pairs, is0_pairs);
          is0_ptr += svcntw();

          for (size_t kt = 0; kt < tiles_per_inner; kt++) {
            auto packed0 = svld1_s8(pg, r0);
            auto rhs0_lo = svmul_s8_x(pg,
                svasr_n_s8_x(pg, svlsl_n_s8_x(pg, packed0, 4), 4), is0_quads);
            auto rhs0_hi = svmul_s8_x(pg,
                svasr_n_s8_x(pg, packed0, 4), is0_quads);

            auto l0 = svld1_s8(pg, lhs_data);
            svmopa_za32_s8_m(0, pg, pg, l0, rhs0_lo);
            lhs_data += 2 * svcntb();

            l0 = svld1_s8(pg, lhs_data);
            svmopa_za32_s8_m(0, pg, pg, l0, rhs0_hi);
            lhs_data += 2 * svcntb();

            r0 += svcntb();
          }
        }

        auto os0 = svld1_f32(pg32, os0_ptr);
        os0_ptr += svcntw();

        for (size_t i = 0; i < m_rows; i++) {
          auto z0 = svcvt_f32_s32_x(pg32,
              svread_hor_za32_s32_m(svdup_n_s32(0), pg32, 0, i));

          if (ob == 0) {
            svst1_f32(np, op + i * p.N, svmul_f32_x(pg32, z0, os0));
          } else {
            svst1_f32(np, op + i * p.N,
                svmla_f32_x(pg32, svld1_f32(np, op + i * p.N), z0, os0));
          }
        }
      }

      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg32,
          svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg32, svld1_f32(pg32, &qp.w_ksums[n]), zp_f32);

      for (size_t i = 0; i < m_rows; i++) {
        auto v0 = svld1_f32(np, op + i * p.N);
        svst1_f32(np, op + i * p.N,
            svmul_f32_x(pg32, svsub_f32_x(pg32, v0, scaled_ksums0), a_scale));
      }

      rhs_col += rhs_n_stride;
      inner_scale_col += num_inner * svcntw();
      outer_scale_col += num_outer * svcntw();
    }

    lhs += svcntb();
  }
}

}  // namespace sme
