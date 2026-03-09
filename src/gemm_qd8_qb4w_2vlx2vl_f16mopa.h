#pragma once

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
