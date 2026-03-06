#include "gemm.h"

// Forward-declare SME kernels without streaming attributes so this TU
// compiles without -march=+sme.  The actual symbols are defined in the
// *_sme.cpp files with the proper __arm_streaming __arm_inout("za").
namespace sme {
void gemm_f32p_f32p_f32_kernel(const GemmParams& p, const void* lhs_packed,
                                const void* rhs_packed, float* out);
void gemm_f16p_f16p_f16_kernel(const GemmParams& p, const void* lhs_packed,
                                const void* rhs_packed, _Float16* out);
void gemm_bf16p_bf16p_bf16_kernel(const GemmParams& p, const void* lhs_packed,
                                const void* rhs_packed, __bf16* out);
void gemm_qd8p_qc8wp_f32_kernel(const GemmParams& p, const void* lhs_packed,
                                const void* rhs_packed, float* out,
                                const QuantParams& qp);
}

namespace sme {

void gemm_f32p_f32p_f32(const GemmParams& p, const void* lhs_packed,
                         const void* rhs_packed, float* out) {
  asm volatile("smstart" ::: "memory");
  gemm_f32p_f32p_f32_kernel(p, lhs_packed, rhs_packed, out);
  asm volatile("smstop" ::: "memory");
}

void gemm_f16p_f16p_f16(const GemmParams& p, const void* lhs_packed,
                         const void* rhs_packed, _Float16* out) {
  asm volatile("smstart" ::: "memory");
  gemm_f16p_f16p_f16_kernel(p, lhs_packed, rhs_packed, out);
  asm volatile("smstop" ::: "memory");
}

void gemm_bf16p_bf16p_bf16(const GemmParams& p, const void* lhs_packed,
                         const void* rhs_packed, __bf16* out) {
  asm volatile("smstart" ::: "memory");
  gemm_bf16p_bf16p_bf16_kernel(p, lhs_packed, rhs_packed, out);
  asm volatile("smstop" ::: "memory");
}

void gemm_qd8p_qc8wp_f32(const GemmParams& p, const void* lhs_packed,
                       const void* rhs_packed, float* out,
                       const QuantParams& qp) {
  asm volatile("smstart" ::: "memory");
  gemm_qd8p_qc8wp_f32_kernel(p, lhs_packed, rhs_packed, out, qp);
  asm volatile("smstop" ::: "memory");
}

}  // namespace sme
