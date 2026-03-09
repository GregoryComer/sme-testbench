#pragma once

#include "gemm.h"

namespace sme {

void gemm_bf16p_bf16p_bf16_2vlx2vl_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    __bf16* out) __arm_streaming __arm_inout("za");

}  // namespace sme
