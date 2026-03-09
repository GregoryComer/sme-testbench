#pragma once

#include "gemm.h"

namespace sme {

void gemm_f16p_f16p_f16_2vlx2vl_kernel(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    _Float16* out) __arm_streaming __arm_inout("za");

}  // namespace sme
