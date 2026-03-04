#pragma once

#include "gemm.h"

#include <cstddef>

namespace sme {

// Returns the packed buffer size in bytes.
size_t packed_size_bytes_f32(size_t rows, size_t cols, size_t tile_r, size_t tile_c);
size_t packed_size_bytes_f32(size_t rows, size_t cols, const PackingParams& params);

// Generic tile packing routine.
//
// Template Args:
//   TransposeInner - transpose elements within each tile
//   TransposeOuter - traverse tiles in transposed order
//
// Args:
//   data    - float[rows, cols], row-major
//   rows    - row count
//   cols    - col count
//   tile_r  - rows per tile
//   tile_c  - cols per tile
//   out     - output buffer (at least packed_size_bytes_f32 bytes)
template <bool TransposeInner, bool TransposeOuter>
void pack_f32(const float* data, size_t rows, size_t cols, size_t tile_r,
              size_t tile_c, void* out);

// Dynamic dispatch wrapper that selects the template instantiation based on
// the transpose flags in PackingParams.
void pack_f32(const float* data, size_t rows, size_t cols,
              const PackingParams& params, void* out);

}  // namespace sme
