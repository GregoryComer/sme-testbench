#pragma once

// Internal header for the SME kernel. Not part of the public API.
// Compiled with -march=armv8-a+sme; do not include from non-SME TUs.

#include "gemm.h"

namespace sme {

void gemm_f32p_f32p_f32_4vlxvl_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out) __arm_streaming __arm_inout("za");

}  // namespace sme
