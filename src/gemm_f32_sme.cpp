#include "gemm_f32_sme.h"

#include <algorithm>
#include <arm_sme.h>

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
      .lhs = {.tile_rows = svl, .tile_cols = 1,
              .transpose_inner = true, .transpose_outer = false},
      .rhs = {.tile_rows = 1, .tile_cols = svl,
              .transpose_inner = false, .transpose_outer = true},
  };
}

void gemm_f32p_f32p_f32_kernel(
    const GemmParams& p, const void* lhs_packed, const void* rhs_packed,
    float* out) __arm_streaming __arm_inout("za") {
  const auto* lhs = static_cast<const float*>(lhs_packed);
  const auto* rhs = static_cast<const float*>(rhs_packed);

  for (auto m = 0u; m < p.M; m += svcntw()) {
    const float* rhs_data = rhs;
    svbool_t m_pred = svwhilelt_b32(m, p.M);

    for (auto n = 0u; n < p.N; n += svcntw()) {
      const float* lhs_data = lhs;
      svbool_t n_pred = svwhilelt_b32(n, p.N);
      svzero_za();

      for (auto k = 0u; k < p.K; k++) {
        svfloat32_t lhs_col0 = svld1_f32(m_pred, lhs_data);
        svfloat32_t rhs_row0 = svld1_f32(n_pred, rhs_data);

        svmopa_za32_f32_m(0, m_pred, n_pred, lhs_col0, rhs_row0);

        lhs_data += svcntw();
        rhs_data += svcntw();
      }

      // Store tile one horizontal slice at a time.
      float* out_data = out + m * p.N + n;
      uint64_t rows = std::min(svcntw(), static_cast<uint64_t>(p.M - m));
      for (uint32_t i = 0; i < rows; i++) {
        svst1_hor_za32(0, i, n_pred, out_data + i * p.N);
      }
    }

    lhs += p.K * svcntw();
  }
}

}  // namespace sme
