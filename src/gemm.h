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

GemmPackingParams gemm_f32_packing_params();

// Tile dimensions for packing, determined by the SME kernel.
size_t gemm_f32_tile_m();
size_t gemm_f32_tile_n();
size_t gemm_f32_tile_k();

// ---------- GEMM kernel -------------------------------------------------------

// Non-streaming entry point. Will manage streaming mode + ZA internally once
// the SME kernel is ready and runtime detection is in place.
void gemm_f32p_f32p_f32(
    const GemmParams& p,
    const void* lhs_packed,
    const void* rhs_packed,
    float* out);
// ---------- Reference ---------------------------------------------------------

void gemm_f32_f32_f32_reference(
  const GemmParams& p, 
  const float* lhs,
  const float* rhs,
  float* output);

}  // namespace sme
