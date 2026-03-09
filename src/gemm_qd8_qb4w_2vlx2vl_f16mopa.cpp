#include "gemm_qd8_qb4w_2vlx2vl_f16mopa.h"

#include <algorithm>
#include <arm_sme.h>

namespace sme {

void gemm_qd8p_qb4wp_f32_2vlx2vl_f16mopa_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams& qp) {

  auto lhs = static_cast<const int8_t*>(lhs_packed);
  auto rhs_base = static_cast<const int8_t*>(rhs_packed);
  const size_t K_pad = (p.K + 3) & ~size_t(3);
  const size_t rhs_n_stride = (K_pad / 8) * svcntb();
  const size_t blocks = p.K / qp.group_size;
  const size_t tiles_per_block = qp.group_size / 8;

  svbool_t pg8 = svptrue_b8();
  svbool_t pg16 = svptrue_b16();
  svbool_t pg32 = svptrue_b32();
  size_t m = 0;

  // --- Full 2x2 body (2 M-subtiles × 2 N-subtiles) --------------------------
  for (; m + svcntw() * 2 <= p.M; m += svcntw() * 2) {
    size_t n = 0;
    auto rhs_col = rhs_base;
    const float* scale_col = qp.w_scales;

    // -- 2x2 N loop --
    for (; n + svcntw() * 2 <= p.N; n += svcntw() * 2) {
      svzero_za();

      auto lhs_data = lhs;
      auto r0 = rhs_col;
      auto r1 = rhs_col + rhs_n_stride;
      auto ws0_ptr = scale_col;
      auto ws1_ptr = scale_col + blocks * svcntw();

      for (size_t block = 0; block < blocks; block++) {
        auto ws0_f32 = svld1_f32(pg32, ws0_ptr);
        auto ws1_f32 = svld1_f32(pg32, ws1_ptr);
        auto ws0_f16 = svcvt_f16_f32_x(pg32, ws0_f32);
        auto ws1_f16 = svcvt_f16_f32_x(pg32, ws1_f32);
        auto ws0_dup = svtrn1_f16(ws0_f16, ws0_f16);
        auto ws1_dup = svtrn1_f16(ws1_f16, ws1_f16);
        ws0_ptr += svcntw();
        ws1_ptr += svcntw();

        for (size_t kt = 0; kt < tiles_per_block; kt++) {
          // Unpack int4 RHS nibbles.
          auto packed0 = svld1_s8(pg8, r0);
          auto rhs0_lo_s8 = svasr_n_s8_x(pg8, svlsl_n_s8_x(pg8, packed0, 4), 4);
          auto rhs0_hi_s8 = svasr_n_s8_x(pg8, packed0, 4);

          auto packed1 = svld1_s8(pg8, r1);
          auto rhs1_lo_s8 = svasr_n_s8_x(pg8, svlsl_n_s8_x(pg8, packed1, 4), 4);
          auto rhs1_hi_s8 = svasr_n_s8_x(pg8, packed1, 4);

          // Convert RHS s8→f16: deinterleaved packing means unpklo=k01, unpkhi=k23.
          auto r0_lo_k01 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpklo_s16(rhs0_lo_s8)), ws0_dup);
          auto r0_lo_k23 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpkhi_s16(rhs0_lo_s8)), ws0_dup);
          auto r0_hi_k01 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpklo_s16(rhs0_hi_s8)), ws0_dup);
          auto r0_hi_k23 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpkhi_s16(rhs0_hi_s8)), ws0_dup);
          auto r1_lo_k01 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpklo_s16(rhs1_lo_s8)), ws1_dup);
          auto r1_lo_k23 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpkhi_s16(rhs1_lo_s8)), ws1_dup);
          auto r1_hi_k01 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpklo_s16(rhs1_hi_s8)), ws1_dup);
          auto r1_hi_k23 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpkhi_s16(rhs1_hi_s8)), ws1_dup);

          // LHS: per-subtile deinterleaved layout [k01|k23] per subtile.
          auto sub0 = svld1_s8(pg8, lhs_data);
          auto l0_k01 = svcvt_f16_s16_x(pg16, svunpklo_s16(sub0));
          auto l0_k23 = svcvt_f16_s16_x(pg16, svunpkhi_s16(sub0));
          auto sub1 = svld1_s8(pg8, lhs_data + svcntb());
          auto l1_k01 = svcvt_f16_s16_x(pg16, svunpklo_s16(sub1));
          auto l1_k23 = svcvt_f16_s16_x(pg16, svunpkhi_s16(sub1));

          // FMOPA for nibble-low K[0..1]:
          svmopa_za32_f16_m(0, pg16, pg16, l0_k01, r0_lo_k01);
          svmopa_za32_f16_m(1, pg16, pg16, l1_k01, r0_lo_k01);
          svmopa_za32_f16_m(2, pg16, pg16, l0_k01, r1_lo_k01);
          svmopa_za32_f16_m(3, pg16, pg16, l1_k01, r1_lo_k01);
          // FMOPA for nibble-low K[2..3]:
          svmopa_za32_f16_m(0, pg16, pg16, l0_k23, r0_lo_k23);
          svmopa_za32_f16_m(1, pg16, pg16, l1_k23, r0_lo_k23);
          svmopa_za32_f16_m(2, pg16, pg16, l0_k23, r1_lo_k23);
          svmopa_za32_f16_m(3, pg16, pg16, l1_k23, r1_lo_k23);
          lhs_data += 2 * svcntb();

          // Second LHS load: K[4..7].
          sub0 = svld1_s8(pg8, lhs_data);
          l0_k01 = svcvt_f16_s16_x(pg16, svunpklo_s16(sub0));
          l0_k23 = svcvt_f16_s16_x(pg16, svunpkhi_s16(sub0));
          sub1 = svld1_s8(pg8, lhs_data + svcntb());
          l1_k01 = svcvt_f16_s16_x(pg16, svunpklo_s16(sub1));
          l1_k23 = svcvt_f16_s16_x(pg16, svunpkhi_s16(sub1));

          // FMOPA for nibble-high K[4..5]:
          svmopa_za32_f16_m(0, pg16, pg16, l0_k01, r0_hi_k01);
          svmopa_za32_f16_m(1, pg16, pg16, l1_k01, r0_hi_k01);
          svmopa_za32_f16_m(2, pg16, pg16, l0_k01, r1_hi_k01);
          svmopa_za32_f16_m(3, pg16, pg16, l1_k01, r1_hi_k01);
          // FMOPA for nibble-high K[6..7]:
          svmopa_za32_f16_m(0, pg16, pg16, l0_k23, r0_hi_k23);
          svmopa_za32_f16_m(1, pg16, pg16, l1_k23, r0_hi_k23);
          svmopa_za32_f16_m(2, pg16, pg16, l0_k23, r1_hi_k23);
          svmopa_za32_f16_m(3, pg16, pg16, l1_k23, r1_hi_k23);
          lhs_data += 2 * svcntb();

          r0 += svcntb();
          r1 += svcntb();
        }
      }

      // Epilogue: read ZA f32 tiles, apply ksums correction and a_scale.
      float* op = out + m * p.N + n;
      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg32, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg32, svld1_f32(pg32, &qp.w_ksums[n]), zp_f32);
      auto scaled_ksums1 = svmul_f32_x(pg32, svld1_f32(pg32, &qp.w_ksums[n + svcntw()]), zp_f32);

      for (uint32_t i = 0; i < svcntw(); i++) {
        auto z0 = svread_hor_za32_f32_m(svdup_n_f32(0), pg32, 0, i);
        auto z1 = svread_hor_za32_f32_m(svdup_n_f32(0), pg32, 1, i);
        auto z2 = svread_hor_za32_f32_m(svdup_n_f32(0), pg32, 2, i);
        auto z3 = svread_hor_za32_f32_m(svdup_n_f32(0), pg32, 3, i);

        svst1_f32(pg32, op + i * p.N,
            svmul_f32_x(pg32, svsub_f32_x(pg32, z0, scaled_ksums0), a_scale));
        svst1_f32(pg32, op + (i + svcntw()) * p.N,
            svmul_f32_x(pg32, svsub_f32_x(pg32, z1, scaled_ksums0), a_scale));
        svst1_f32(pg32, op + i * p.N + svcntw(),
            svmul_f32_x(pg32, svsub_f32_x(pg32, z2, scaled_ksums1), a_scale));
        svst1_f32(pg32, op + (i + svcntw()) * p.N + svcntw(),
            svmul_f32_x(pg32, svsub_f32_x(pg32, z3, scaled_ksums1), a_scale));
      }

      rhs_col += 2 * rhs_n_stride;
      scale_col += 2 * blocks * svcntw();
    }

    // -- 2x1 N-tail (remaining columns after 2x2 body) --
    for (; n < p.N; n += svcntw()) {
      svzero_za();

      auto lhs_data = lhs;
      auto r0 = rhs_col;
      auto ws0_ptr = scale_col;
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);

      for (size_t block = 0; block < blocks; block++) {
        auto ws0_f32 = svld1_f32(pg32, ws0_ptr);
        auto ws0_f16 = svcvt_f16_f32_x(pg32, ws0_f32);
        auto ws0_dup = svtrn1_f16(ws0_f16, ws0_f16);
        ws0_ptr += svcntw();

        for (size_t kt = 0; kt < tiles_per_block; kt++) {
          auto packed0 = svld1_s8(pg8, r0);
          auto rhs0_lo_s8 = svasr_n_s8_x(pg8, svlsl_n_s8_x(pg8, packed0, 4), 4);
          auto rhs0_hi_s8 = svasr_n_s8_x(pg8, packed0, 4);

          auto r0_lo_k01 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpklo_s16(rhs0_lo_s8)), ws0_dup);
          auto r0_lo_k23 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpkhi_s16(rhs0_lo_s8)), ws0_dup);
          auto r0_hi_k01 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpklo_s16(rhs0_hi_s8)), ws0_dup);
          auto r0_hi_k23 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpkhi_s16(rhs0_hi_s8)), ws0_dup);

          // LHS: per-subtile layout, one load gives both k01 and k23.
          auto sub0 = svld1_s8(pg8, lhs_data);
          auto l0_k01 = svcvt_f16_s16_x(pg16, svunpklo_s16(sub0));
          auto l0_k23 = svcvt_f16_s16_x(pg16, svunpkhi_s16(sub0));
          auto sub1 = svld1_s8(pg8, lhs_data + svcntb());
          auto l1_k01 = svcvt_f16_s16_x(pg16, svunpklo_s16(sub1));
          auto l1_k23 = svcvt_f16_s16_x(pg16, svunpkhi_s16(sub1));

          svmopa_za32_f16_m(0, pg16, pg16, l0_k01, r0_lo_k01);
          svmopa_za32_f16_m(0, pg16, pg16, l0_k23, r0_lo_k23);
          svmopa_za32_f16_m(1, pg16, pg16, l1_k01, r0_lo_k01);
          svmopa_za32_f16_m(1, pg16, pg16, l1_k23, r0_lo_k23);
          lhs_data += 2 * svcntb();

          sub0 = svld1_s8(pg8, lhs_data);
          l0_k01 = svcvt_f16_s16_x(pg16, svunpklo_s16(sub0));
          l0_k23 = svcvt_f16_s16_x(pg16, svunpkhi_s16(sub0));
          sub1 = svld1_s8(pg8, lhs_data + svcntb());
          l1_k01 = svcvt_f16_s16_x(pg16, svunpklo_s16(sub1));
          l1_k23 = svcvt_f16_s16_x(pg16, svunpkhi_s16(sub1));

          svmopa_za32_f16_m(0, pg16, pg16, l0_k01, r0_hi_k01);
          svmopa_za32_f16_m(0, pg16, pg16, l0_k23, r0_hi_k23);
          svmopa_za32_f16_m(1, pg16, pg16, l1_k01, r0_hi_k01);
          svmopa_za32_f16_m(1, pg16, pg16, l1_k23, r0_hi_k23);
          lhs_data += 2 * svcntb();

          r0 += svcntb();
        }
      }

      // Epilogue for 2x1 N-tail.
      float* op = out + m * p.N + n;
      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg32, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg32, svld1_f32(pg32, &qp.w_ksums[n]), zp_f32);

      for (uint32_t i = 0; i < svcntw(); i++) {
        auto z0 = svread_hor_za32_f32_m(svdup_n_f32(0), pg32, 0, i);
        auto z1 = svread_hor_za32_f32_m(svdup_n_f32(0), pg32, 1, i);

        svst1_f32(np, op + i * p.N,
            svmul_f32_x(pg32, svsub_f32_x(pg32, z0, scaled_ksums0), a_scale));
        svst1_f32(np, op + (i + svcntw()) * p.N,
            svmul_f32_x(pg32, svsub_f32_x(pg32, z1, scaled_ksums0), a_scale));
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
      svzero_za();

      auto lhs_data = lhs;
      auto r0 = rhs_col;
      auto ws0_ptr = scale_col;
      svbool_t np = svwhilelt_b32(n, (uint64_t)p.N);

      for (size_t block = 0; block < blocks; block++) {
        auto ws0_f32 = svld1_f32(pg32, ws0_ptr);
        auto ws0_f16 = svcvt_f16_f32_x(pg32, ws0_f32);
        auto ws0_dup = svtrn1_f16(ws0_f16, ws0_f16);
        ws0_ptr += svcntw();

        for (size_t kt = 0; kt < tiles_per_block; kt++) {
          auto packed0 = svld1_s8(pg8, r0);
          auto rhs0_lo_s8 = svasr_n_s8_x(pg8, svlsl_n_s8_x(pg8, packed0, 4), 4);
          auto rhs0_hi_s8 = svasr_n_s8_x(pg8, packed0, 4);

          auto r0_lo_k01 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpklo_s16(rhs0_lo_s8)), ws0_dup);
          auto r0_lo_k23 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpkhi_s16(rhs0_lo_s8)), ws0_dup);
          auto r0_hi_k01 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpklo_s16(rhs0_hi_s8)), ws0_dup);
          auto r0_hi_k23 = svmul_f16_x(pg16, svcvt_f16_s16_x(pg16, svunpkhi_s16(rhs0_hi_s8)), ws0_dup);

          // M-tail: one load per subtile gives both k01 and k23.
          auto sub0 = svld1_s8(pg8, lhs_data);
          auto l0_k01 = svcvt_f16_s16_x(pg16, svunpklo_s16(sub0));
          auto l0_k23 = svcvt_f16_s16_x(pg16, svunpkhi_s16(sub0));

          svmopa_za32_f16_m(0, pg16, pg16, l0_k01, r0_lo_k01);
          svmopa_za32_f16_m(0, pg16, pg16, l0_k23, r0_lo_k23);
          lhs_data += 2 * svcntb();

          sub0 = svld1_s8(pg8, lhs_data);
          l0_k01 = svcvt_f16_s16_x(pg16, svunpklo_s16(sub0));
          l0_k23 = svcvt_f16_s16_x(pg16, svunpkhi_s16(sub0));

          svmopa_za32_f16_m(0, pg16, pg16, l0_k01, r0_hi_k01);
          svmopa_za32_f16_m(0, pg16, pg16, l0_k23, r0_hi_k23);
          lhs_data += 2 * svcntb();

          r0 += svcntb();
        }
      }

      // Epilogue for M-tail.
      float* op = out + m * p.N + n;
      auto a_scale = svdup_n_f32(qp.a_scale);
      auto zp_f32 = svcvt_f32_s32_x(pg32, svdup_n_s32(static_cast<int32_t>(qp.a_zero_point)));
      auto scaled_ksums0 = svmul_f32_x(pg32, svld1_f32(pg32, &qp.w_ksums[n]), zp_f32);

      for (size_t i = 0; i < m_rows; i++) {
        auto z0 = svread_hor_za32_f32_m(svdup_n_f32(0), pg32, 0, i);
        svst1_f32(np, op + i * p.N,
            svmul_f32_x(pg32, svsub_f32_x(pg32, z0, scaled_ksums0), a_scale));
      }

      rhs_col += rhs_n_stride;
      scale_col += blocks * svcntw();
    }

    lhs += svcntb();
  }
}

}  // namespace sme
