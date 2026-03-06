#pragma once

#include <cstddef>
#include <cstdint>

namespace sme {

struct GemmParams {
  size_t M;
  size_t N;
  size_t K;
};

struct PackingParams {
  size_t tile_rows;
  size_t tile_cols;
  bool transpose_inner;
  bool transpose_outer;
};

struct GemmPackingParams {
  PackingParams lhs;
  PackingParams rhs;
};

// ---------- f32 --------------------------------------------------------------

GemmPackingParams gemm_f32_packing_params();

size_t gemm_f32_tile_m();
size_t gemm_f32_tile_n();
size_t gemm_f32_tile_k();

void gemm_f32p_f32p_f32(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out);

void gemm_f32_f32_f32_reference(
    const GemmParams& p,
    const float* lhs,
    const float* rhs,
    float* output);

// ---------- f16 --------------------------------------------------------------

GemmPackingParams gemm_f16_packing_params();

void gemm_f16p_f16p_f16(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    _Float16* out);

void gemm_f16_f16_f16_reference(
    const GemmParams& p,
    const _Float16* lhs,
    const _Float16* rhs,
    _Float16* output);

// ---------- bf16 -------------------------------------------------------------

GemmPackingParams gemm_bf16_packing_params();

void gemm_bf16p_bf16p_bf16(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    __bf16* out);

void gemm_bf16_bf16_bf16_reference(
    const GemmParams& p,
    const __bf16* lhs,
    const __bf16* rhs,
    __bf16* output);

// ---------- qd8×qc8w→f32 (asymmetric activations, per-channel weights) ------

struct QuantParams {
  int8_t a_zero_point;
  float a_scale;
  const float* w_scales;  // per-channel, N elements
  const float* w_ksums;   // precomputed: sum_k(w[k,n]) * w_scales[n], N elements
};

GemmPackingParams gemm_qd8_qc8w_packing_params();

void gemm_qd8p_qc8wp_f32(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const QuantParams& qp);

void gemm_qd8_qc8w_f32_reference(
    const GemmParams& p,
    const int8_t* lhs,
    const int8_t* rhs,
    float* output,
    const QuantParams& qp);

}  // namespace sme
