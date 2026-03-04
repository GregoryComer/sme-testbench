#include "pack.h"

#include <algorithm>

namespace sme {

size_t packed_size_bytes_f32(size_t rows, size_t cols, size_t tile_r,
                       size_t tile_c) {
  // We can ignore transposition here since the size is the same either way.
  size_t out_row_tiles = (rows + tile_r - 1) / tile_r;
  size_t out_col_tiles = (cols + tile_c - 1) / tile_c;

  return out_row_tiles * out_col_tiles * tile_r * tile_c * sizeof(float);
}

size_t packed_size_bytes_f32(size_t rows, size_t cols,
                             const PackingParams& params) {
  return packed_size_bytes_f32(rows, cols, params.tile_rows, params.tile_cols);
}

template <bool TransposeInner, bool TransposeOuter>
void pack_f32(const float* data, size_t rows, size_t cols, size_t tile_r,
              size_t tile_c, void* out) {
  size_t out_row_tiles = (rows + tile_r - 1) / tile_r;
  size_t out_col_tiles = (cols + tile_c - 1) / tile_c;
  if constexpr (TransposeOuter) {
    std::swap(out_row_tiles, out_col_tiles);
  }

  // Tile sizes are relative to the input. Compute output tile size -
  // transpose if needed.
  size_t out_tile_r = tile_r;
  size_t out_tile_c = tile_c;
  if constexpr (TransposeInner) {
    std::swap(out_tile_r, out_tile_c);
  }

  float* out_data = static_cast<float*>(out);

  // Packing algorithm - writes output data contiguously. For each output
  // position, find the input position and copy the element.

  // Loop over output tiles.
  for (size_t r_tile = 0u; r_tile < out_row_tiles; r_tile++) {
    for (size_t c_tile = 0u; c_tile < out_col_tiles; c_tile++) {

      // Loop over each element within the output tile.
      for (size_t r_offset = 0u; r_offset < out_tile_r; r_offset++) {
        for (size_t c_offset = 0u; c_offset < out_tile_c; c_offset++) {

          // Load input element.
          size_t input_r_offset = TransposeInner ? c_offset : r_offset;
          size_t input_c_offset = TransposeInner ? r_offset : c_offset;

          size_t input_row, input_col;
          if constexpr (TransposeOuter) {
            input_row = c_tile * tile_r + input_r_offset;
            input_col = r_tile * tile_c + input_c_offset;
          } else {
            input_row = r_tile * tile_r + input_r_offset;
            input_col = c_tile * tile_c + input_c_offset;
          }

          float input_element = 0.0f;
          if (input_row < rows && input_col < cols) {
            input_element = data[input_row * cols + input_col];
          }

          *out_data = input_element;
          out_data++;
        }
      }

    }
  }
}

template void pack_f32<false, false>(const float*, size_t, size_t, size_t,
                                     size_t, void*);
template void pack_f32<false, true>(const float*, size_t, size_t, size_t,
                                    size_t, void*);
template void pack_f32<true, false>(const float*, size_t, size_t, size_t,
                                    size_t, void*);
template void pack_f32<true, true>(const float*, size_t, size_t, size_t,
                                   size_t, void*);

void pack_f32(const float* data, size_t rows, size_t cols,
              const PackingParams& params, void* out) {
  bool ti = params.transpose_inner;
  bool to = params.transpose_outer;
  if (ti && to)
    pack_f32<true, true>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else if (ti)
    pack_f32<true, false>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else if (to)
    pack_f32<false, true>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else
    pack_f32<false, false>(data, rows, cols, params.tile_rows, params.tile_cols, out);
}

}  // namespace sme
