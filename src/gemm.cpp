#include "gemm.h"

// Forward-declare the SME kernel without streaming attributes so this TU
// compiles without -march=+sme.  The actual symbol is defined in
// gemm_f32_sme.cpp with the proper __arm_streaming __arm_inout("za").
namespace sme {
void gemm_f32p_f32p_f32_kernel(const GemmParams& p, const void* lhs_packed,
                                const void* rhs_packed, float* out);
}

namespace sme {

void gemm_f32p_f32p_f32(const GemmParams& p, const void* lhs_packed,
                         const void* rhs_packed, float* out) {
  // Enter streaming mode and enable ZA.
  asm volatile("smstart" ::: "memory");

  gemm_f32p_f32p_f32_kernel(p, lhs_packed, rhs_packed, out);

  // Exit streaming mode and release ZA.
  asm volatile("smstop" ::: "memory");
}

}  // namespace sme
