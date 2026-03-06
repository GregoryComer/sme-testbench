#include "gemm_qd8_qb4w_4vlxvl_sme.h"

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
    const GemmParams& p, const void* /*lhs_packed*/, const void* /*rhs_packed*/,
    float* out, const BlockQuantParams& /*qp*/) __arm_streaming __arm_inout("za") {
  // TODO: implement groupwise 4-bit weight GEMM kernel
  std::memset(out, 0, p.M * p.N * sizeof(float));
}

}  // namespace sme
