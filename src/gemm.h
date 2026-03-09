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

GemmPackingParams gemm_f32_4vlxvl_packing_params();
GemmPackingParams gemm_f32_2vlx2vl_packing_params();

void gemm_f32p_f32p_f32_4vlxvl(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out);

void gemm_f32p_f32p_f32_2vlx2vl(
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

GemmPackingParams gemm_f16_4vlxvl_packing_params();
GemmPackingParams gemm_f16_2vlx2vl_packing_params();

void gemm_f16p_f16p_f16_4vlxvl(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    _Float16* out);

void gemm_f16p_f16p_f16_2vlx2vl(
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

GemmPackingParams gemm_bf16_4vlxvl_packing_params();
GemmPackingParams gemm_bf16_2vlx2vl_packing_params();

void gemm_bf16p_bf16p_bf16_4vlxvl(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    __bf16* out);

void gemm_bf16p_bf16p_bf16_2vlx2vl(
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

GemmPackingParams gemm_qd8_qc8w_4vlxvl_packing_params();
GemmPackingParams gemm_qd8_qc8w_2vlx2vl_packing_params();

void gemm_qd8p_qc8wp_f32_4vlxvl(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const QuantParams& qp);

void gemm_qd8p_qc8wp_f32_2vlx2vl(
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

// ---------- qd8×qc4w→f32 (asymmetric activations, 4-bit per-channel weights)

GemmPackingParams gemm_qd8_qc4w_4vlxvl_packing_params();

void gemm_qd8p_qc4wp_f32_4vlxvl(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const QuantParams& qp);

void gemm_qd8_qc4w_f32_reference(
    const GemmParams& p,
    const int8_t* lhs,
    const int8_t* rhs,
    float* output,
    const QuantParams& qp);

// ---------- qd8×qb4w→f32 (asymmetric activations, 4-bit groupwise weights) --

struct BlockQuantParams {
  int8_t a_zero_point;
  float a_scale;
  size_t group_size;
  const float* w_scales;  // num_groups × N, row-major: w_scales[g * N + n]
  const float* w_ksums;   // N elements, precomputed across all groups
};

GemmPackingParams gemm_qd8_qb4w_4vlxvl_packing_params();
GemmPackingParams gemm_qd8_qb4w_2vlx2vl_packing_params();

void gemm_qd8p_qb4wp_f32_4vlxvl(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams& qp,
    float* scratch);

void gemm_qd8p_qb4wp_f32_2vlx2vl(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams& qp);

GemmPackingParams gemm_qd8_qb4w_2vlxvl_packing_params();

void gemm_qd8p_qb4wp_f32_2vlxvl(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams& qp);

void gemm_qd8p_qb4wp_f32_2vlx2vl_f16mopa(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams& qp);

void gemm_qd8_qb4w_f32_reference(
    const GemmParams& p,
    const int8_t* lhs,
    const int8_t* rhs,
    float* output,
    const BlockQuantParams& qp);

// ---------- qd8×qb4w→f32 two-level block scales (int8 inner + f32 outer) ----
// Two-level quantization where weights are scaled by int8 inner_scale per
// inner block, with int32 accumulation within outer blocks, and f32 outer_scale
// conversion deferred to the (less frequent) outer-block boundary.

struct BlockQuantParams2L {
  int8_t a_zero_point;
  float a_scale;
  size_t inner_group_size;   // e.g. 32, 128
  size_t outer_group_size;   // e.g. 128..4096, must be multiple of inner
  const int8_t* inner_scales;  // tile-packed: [N/tile_n][num_inner][tile_n]
  const float* outer_scales;   // tile-packed: [N/tile_n][num_outer][tile_n]
  const float* w_ksums;        // N elements, precomputed across all groups
};

GemmPackingParams gemm_qd8_qb4w2l_2vlx2vl_packing_params();

void gemm_qd8p_qb4w2lp_f32_2vlx2vl(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out,
    const BlockQuantParams2L& qp);

void gemm_qd8_qb4w2l_f32_reference(
    const GemmParams& p,
    const int8_t* lhs,
    const int8_t* rhs,
    float* output,
    const BlockQuantParams2L& qp);

}  // namespace sme
