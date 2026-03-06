#pragma once

// Internal header for the qd8×qc8w→f32 SME kernel. Not part of the public API.
// Compiled with -march=armv8-a+sme; do not include from non-SME TUs.

#include "gemm.h"

namespace sme {

void gemm_qd8p_qc8wp_f32_4vlxvl_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const QuantParams& qp) __arm_streaming __arm_inout("za");

}  // namespace sme
