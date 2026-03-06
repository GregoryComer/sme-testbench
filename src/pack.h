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

}  // namespace sme
