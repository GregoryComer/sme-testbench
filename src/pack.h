#pragma once

#include "gemm.h"

#include <cstddef>
#include <cstdint>

namespace sme {

// ---------- f32 --------------------------------------------------------------

size_t packed_size_bytes_f32(size_t rows, size_t cols, size_t tile_r, size_t tile_c);
size_t packed_size_bytes_f32(size_t rows, size_t cols, const PackingParams& params);

template <bool TransposeInner, bool TransposeOuter>
void pack_f32(const float* data, size_t rows, size_t cols, size_t tile_r,
              size_t tile_c, void* out);

void pack_f32(const float* data, size_t rows, size_t cols,
              const PackingParams& params, void* out);

// ---------- f16 --------------------------------------------------------------

size_t packed_size_bytes_f16(size_t rows, size_t cols, size_t tile_r, size_t tile_c);
size_t packed_size_bytes_f16(size_t rows, size_t cols, const PackingParams& params);

template <bool TransposeInner, bool TransposeOuter>
void pack_f16(const _Float16* data, size_t rows, size_t cols, size_t tile_r,
              size_t tile_c, void* out);

void pack_f16(const _Float16* data, size_t rows, size_t cols,
              const PackingParams& params, void* out);

// ---------- bf16 -------------------------------------------------------------

size_t packed_size_bytes_bf16(size_t rows, size_t cols, size_t tile_r, size_t tile_c);
size_t packed_size_bytes_bf16(size_t rows, size_t cols, const PackingParams& params);

template <bool TransposeInner, bool TransposeOuter>
void pack_bf16(const __bf16* data, size_t rows, size_t cols, size_t tile_r,
              size_t tile_c, void* out);

void pack_bf16(const __bf16* data, size_t rows, size_t cols,
              const PackingParams& params, void* out);

// ---------- u8 (unsigned 8-bit) -----------------------------------------------

size_t packed_size_bytes_u8(size_t rows, size_t cols, size_t tile_r, size_t tile_c);
size_t packed_size_bytes_u8(size_t rows, size_t cols, const PackingParams& params);

template <bool TransposeInner, bool TransposeOuter>
void pack_u8(const uint8_t* data, size_t rows, size_t cols, size_t tile_r,
             size_t tile_c, void* out);

void pack_u8(const uint8_t* data, size_t rows, size_t cols,
             const PackingParams& params, void* out);

// ---------- s8 (signed 8-bit) ------------------------------------------------

size_t packed_size_bytes_s8(size_t rows, size_t cols, size_t tile_r, size_t tile_c);
size_t packed_size_bytes_s8(size_t rows, size_t cols, const PackingParams& params);

template <bool TransposeInner, bool TransposeOuter>
void pack_s8(const int8_t* data, size_t rows, size_t cols, size_t tile_r,
             size_t tile_c, void* out);

void pack_s8(const int8_t* data, size_t rows, size_t cols,
             const PackingParams& params, void* out);

void compute_ksums_s8(const int8_t* data, size_t K, size_t N, const float* w_scales,
    float* out);

void compute_group_ksums_s8(const int8_t* data, size_t K, size_t N,
    size_t group_size, const float* w_scales, float* out);

// ---------- group scale tiling -----------------------------------------------
// Tile-packs per-group weight scales from row-major [num_groups][N] into
// [N/tile_n][num_groups][tile_n] so that all group scales for one N-tile are
// contiguous.

size_t packed_group_scales_len(size_t num_groups, size_t N, size_t tile_n);

void pack_group_scales(const float* scales, size_t num_groups, size_t N,
                       size_t tile_n, float* out);

// ---------- s4 (signed 4-bit, nibble-packed) ---------------------------------
// Input is int8_t* with values in [-8, 7]. Output is nibble-packed tiled data.

size_t packed_size_bytes_s4(size_t rows, size_t cols, size_t tile_r, size_t tile_c);
size_t packed_size_bytes_s4(size_t rows, size_t cols, const PackingParams& params);

void pack_s4(const int8_t* data, size_t rows, size_t cols,
             const PackingParams& params, void* out);

}  // namespace sme
