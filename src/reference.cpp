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

void gemm_qd8_qc4w_f32_reference(const GemmParams& p, const int8_t* lhs,
                                  const int8_t* rhs, float* output,
                                  const QuantParams& qp) {
  // Identical to qc8w — same per-channel dequantisation, just smaller weight range.
  gemm_qd8_qc8w_f32_reference(p, lhs, rhs, output, qp);
}

void gemm_qd8_qb4w_f32_reference(const GemmParams& p, const int8_t* lhs,
                                  const int8_t* rhs, float* output,
                                  const BlockQuantParams& qp) {
  size_t num_groups = (p.K + qp.group_size - 1) / qp.group_size;
  for (size_t m = 0; m < p.M; ++m) {
    for (size_t n = 0; n < p.N; ++n) {
      float acc = 0.0f;
      for (size_t g = 0; g < num_groups; ++g) {
        size_t k_start = g * qp.group_size;
        size_t k_end = k_start + qp.group_size;
        if (k_end > p.K) k_end = p.K;
        int32_t group_acc = 0;
        for (size_t k = k_start; k < k_end; ++k) {
          group_acc += (static_cast<int32_t>(lhs[m * p.K + k]) -
                        static_cast<int32_t>(qp.a_zero_point)) *
                       static_cast<int32_t>(rhs[k * p.N + n]);
        }
        acc += qp.w_scales[g * p.N + n] * static_cast<float>(group_acc);
      }
      output[m * p.N + n] = qp.a_scale * acc;
    }
  }
}

void gemm_qd8_qb4w2l_f32_reference(const GemmParams& p, const int8_t* lhs,
                                    const int8_t* rhs, float* output,
                                    const BlockQuantParams2L& qp) {
  size_t num_inner = (p.K + qp.inner_group_size - 1) / qp.inner_group_size;
  size_t inner_per_outer = qp.outer_group_size / qp.inner_group_size;
  for (size_t m = 0; m < p.M; ++m) {
    for (size_t n = 0; n < p.N; ++n) {
      float acc = 0.0f;
      for (size_t ib = 0; ib < num_inner; ++ib) {
        size_t ob = ib / inner_per_outer;
        size_t k_start = ib * qp.inner_group_size;
        size_t k_end = k_start + qp.inner_group_size;
        if (k_end > p.K) k_end = p.K;
        int32_t block_acc = 0;
        for (size_t k = k_start; k < k_end; ++k) {
          block_acc += (static_cast<int32_t>(lhs[m * p.K + k]) -
                        static_cast<int32_t>(qp.a_zero_point)) *
                       static_cast<int32_t>(rhs[k * p.N + n]);
        }
        float effective_scale = qp.outer_scales[ob * p.N + n] *
                                static_cast<float>(qp.inner_scales[ib * p.N + n]);
        acc += effective_scale * static_cast<float>(block_acc);
      }
      output[m * p.N + n] = qp.a_scale * acc;
    }
  }
}

}  // namespace sme
