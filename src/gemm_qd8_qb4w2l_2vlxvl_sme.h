#pragma once

// Internal header for the 2x2 qd8×qb4w→f32 SME kernel with 2-level block scales.
// Pre-multiplies expanded int4 weights by int8 inner scale before SMOPA.
// All 4 ZA tiles used for int32 SMOPA accumulation (2M × 2N).
// Outer f32 accumulator spills to the output buffer.
// Compiled with -march=armv8-a+sme; do not include from non-SME TUs.

#include "gemm.h"

namespace sme {

GemmPackingParams gemm_qd8_qb4w2l_2vlxvl_packing_params();

void gemm_qd8p_qb4w2lp_f32_2vlxvl_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams2L& qp) __arm_streaming __arm_inout("za");

}  // namespace sme
