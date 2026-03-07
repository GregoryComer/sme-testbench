#pragma once

// Internal header for the 2x2 qd8×qb4w→f32 SME kernel. Not part of the public API.
// Compiled with -march=armv8-a+sme; do not include from non-SME TUs.

#include "gemm.h"

namespace sme {

GemmPackingParams gemm_qd8_qb4w_2vlx2vl_packing_params();

void gemm_qd8p_qb4wp_f32_2vlx2vl_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams& qp) __arm_streaming __arm_inout("za");

}  // namespace sme
