#pragma once

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
