#pragma once

// Internal header for the f16 SME kernel. Not part of the public API.
// Compiled with -march=armv8-a+sme; do not include from non-SME TUs.

#include "gemm.h"

namespace sme {

void gemm_f16p_f16p_f16_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    _Float16* out) __arm_streaming __arm_inout("za");

}  // namespace sme
