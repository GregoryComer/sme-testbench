#pragma once

#include <cstddef>

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

}  // namespace sme
