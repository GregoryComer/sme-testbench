#pragma once

// Internal header for the 2x1 qd8×qb4w→f32 SME kernel with ZA float accumulation.
// Uses ZA0/ZA1 for int32 SMOPA and ZA2/ZA3 for float32 cross-group accumulation.
// Compiled with -march=armv8-a+sme; do not include from non-SME TUs.

#include "gemm.h"

namespace sme {

GemmPackingParams gemm_qd8_qb4w_2vlxvl_packing_params();

void gemm_qd8p_qb4wp_f32_2vlxvl_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams& qp) __arm_streaming __arm_inout("za");

}  // namespace sme
