#pragma once

#include "gemm.h"

namespace sme {

void gemm_f32p_f32p_f32_4vlxvl_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out) __arm_streaming __arm_inout("za");

}  // namespace sme
