#include "gemm.h"

namespace sme {

void gemm_f32_f32_f32_reference(const GemmParams& p, const float* lhs,
                                const float* rhs, float* output) {
  for (size_t m = 0; m < p.M; ++m) {
    for (size_t n = 0; n < p.N; ++n) {
      float acc = 0.0f;
      for (size_t k = 0; k < p.K; ++k)
        acc += lhs[m * p.K + k] * rhs[k * p.N + n];
      output[m * p.N + n] = acc;
    }
  }
}

}  // namespace sme
