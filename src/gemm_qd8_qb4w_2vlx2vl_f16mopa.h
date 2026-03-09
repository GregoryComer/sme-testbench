#pragma once

// Internal header for the 2x2 qd8×qb4w→f32 kernel using f16 widening FMOPA.
// Converts int8 activations and int4 weights to f16 on the fly, pre-multiplies
// weights by their per-group scale (as f16), and uses svmopa_za32_f16_m
// (f16→f32 widening FMOPA, rank-2) to accumulate directly into f32 ZA tiles
// across all groups with no per-group ZA manipulation.
// Compiled with -march=armv8-a+sme; do not include from non-SME TUs.

#include "gemm.h"

namespace sme {

__arm_locally_streaming __arm_new("za")
void gemm_qd8p_qb4wp_f32_2vlx2vl_f16mopa_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams& qp);

}  // namespace sme
