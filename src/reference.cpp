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

void gemm_f16_f16_f16_reference(const GemmParams& p, const _Float16* lhs,
                                const _Float16* rhs, _Float16* output) {
  for (size_t m = 0; m < p.M; ++m) {
    for (size_t n = 0; n < p.N; ++n) {
      float acc = 0.0f;
      for (size_t k = 0; k < p.K; ++k)
        acc += static_cast<float>(lhs[m * p.K + k]) *
               static_cast<float>(rhs[k * p.N + n]);
      output[m * p.N + n] = static_cast<_Float16>(acc);
    }
  }
}

void gemm_bf16_bf16_bf16_reference(const GemmParams& p, const __bf16* lhs,
                                const __bf16* rhs, __bf16* output) {
  for (size_t m = 0; m < p.M; ++m) {
    for (size_t n = 0; n < p.N; ++n) {
      float acc = 0.0f;
      for (size_t k = 0; k < p.K; ++k)
        acc += static_cast<float>(lhs[m * p.K + k]) *
               static_cast<float>(rhs[k * p.N + n]);
      output[m * p.N + n] = static_cast<__bf16>(acc);
    }
  }
}

void gemm_qd8_qc8w_f32_reference(const GemmParams& p, const int8_t* lhs,
                               const int8_t* rhs, float* output,
                               const QuantParams& qp) {
  for (size_t m = 0; m < p.M; ++m) {
    for (size_t n = 0; n < p.N; ++n) {
      int32_t acc = 0;
      for (size_t k = 0; k < p.K; ++k)
        acc += (static_cast<int32_t>(lhs[m * p.K + k]) -
                static_cast<int32_t>(qp.a_zero_point)) *
               static_cast<int32_t>(rhs[k * p.N + n]);
      output[m * p.N + n] = qp.a_scale * qp.w_scales[n] *
                             static_cast<float>(acc);
    }
  }
}

}  // namespace sme
