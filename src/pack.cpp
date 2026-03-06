#include "pack.h"

#include <algorithm>

namespace sme {
namespace detail {

template <typename T>
size_t packed_size_bytes(size_t rows, size_t cols, size_t tile_r,
                         size_t tile_c) {
  size_t out_row_tiles = (rows + tile_r - 1) / tile_r;
  size_t out_col_tiles = (cols + tile_c - 1) / tile_c;
  return out_row_tiles * out_col_tiles * tile_r * tile_c * sizeof(T);
}

template <typename T, bool TransposeInner, bool TransposeOuter>
void pack(const T* data, size_t rows, size_t cols, size_t tile_r,
          size_t tile_c, void* out) {
  size_t out_row_tiles = (rows + tile_r - 1) / tile_r;
  size_t out_col_tiles = (cols + tile_c - 1) / tile_c;
  if constexpr (TransposeOuter) {
    std::swap(out_row_tiles, out_col_tiles);
  }

  size_t out_tile_r = tile_r;
  size_t out_tile_c = tile_c;
  if constexpr (TransposeInner) {
    std::swap(out_tile_r, out_tile_c);
  }

  T* out_data = static_cast<T*>(out);

  for (size_t r_tile = 0u; r_tile < out_row_tiles; r_tile++) {
    for (size_t c_tile = 0u; c_tile < out_col_tiles; c_tile++) {
      for (size_t r_offset = 0u; r_offset < out_tile_r; r_offset++) {
        for (size_t c_offset = 0u; c_offset < out_tile_c; c_offset++) {
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

          T input_element = static_cast<T>(0);
          if (input_row < rows && input_col < cols) {
            input_element = data[input_row * cols + input_col];
          }

          *out_data++ = input_element;
        }
      }
    }
  }
}

}  // namespace detail

// ---------- f32 --------------------------------------------------------------

size_t packed_size_bytes_f32(size_t rows, size_t cols, size_t tile_r,
                             size_t tile_c) {
  return detail::packed_size_bytes<float>(rows, cols, tile_r, tile_c);
}

size_t packed_size_bytes_f32(size_t rows, size_t cols,
                             const PackingParams& params) {
  return packed_size_bytes_f32(rows, cols, params.tile_rows, params.tile_cols);
}

template <bool TransposeInner, bool TransposeOuter>
void pack_f32(const float* data, size_t rows, size_t cols, size_t tile_r,
              size_t tile_c, void* out) {
  detail::pack<float, TransposeInner, TransposeOuter>(data, rows, cols, tile_r,
                                                       tile_c, out);
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

// ---------- f16 --------------------------------------------------------------

size_t packed_size_bytes_f16(size_t rows, size_t cols, size_t tile_r,
                             size_t tile_c) {
  return detail::packed_size_bytes<_Float16>(rows, cols, tile_r, tile_c);
}

size_t packed_size_bytes_f16(size_t rows, size_t cols,
                             const PackingParams& params) {
  return packed_size_bytes_f16(rows, cols, params.tile_rows, params.tile_cols);
}

template <bool TransposeInner, bool TransposeOuter>
void pack_f16(const _Float16* data, size_t rows, size_t cols, size_t tile_r,
              size_t tile_c, void* out) {
  detail::pack<_Float16, TransposeInner, TransposeOuter>(data, rows, cols,
                                                          tile_r, tile_c, out);
}

template void pack_f16<false, false>(const _Float16*, size_t, size_t, size_t,
                                     size_t, void*);
template void pack_f16<false, true>(const _Float16*, size_t, size_t, size_t,
                                    size_t, void*);
template void pack_f16<true, false>(const _Float16*, size_t, size_t, size_t,
                                    size_t, void*);
template void pack_f16<true, true>(const _Float16*, size_t, size_t, size_t,
                                   size_t, void*);

void pack_f16(const _Float16* data, size_t rows, size_t cols,
              const PackingParams& params, void* out) {
  bool ti = params.transpose_inner;
  bool to = params.transpose_outer;
  if (ti && to)
    pack_f16<true, true>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else if (ti)
    pack_f16<true, false>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else if (to)
    pack_f16<false, true>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else
    pack_f16<false, false>(data, rows, cols, params.tile_rows, params.tile_cols, out);
}

// ---------- bf16 -------------------------------------------------------------

size_t packed_size_bytes_bf16(size_t rows, size_t cols, size_t tile_r,
                             size_t tile_c) {
  return detail::packed_size_bytes<__bf16>(rows, cols, tile_r, tile_c);
}

size_t packed_size_bytes_bf16(size_t rows, size_t cols,
                             const PackingParams& params) {
  return packed_size_bytes_bf16(rows, cols, params.tile_rows, params.tile_cols);
}

template <bool TransposeInner, bool TransposeOuter>
void pack_bf16(const __bf16* data, size_t rows, size_t cols, size_t tile_r,
              size_t tile_c, void* out) {
  detail::pack<__bf16, TransposeInner, TransposeOuter>(data, rows, cols,
                                                          tile_r, tile_c, out);
}

template void pack_bf16<false, false>(const __bf16*, size_t, size_t, size_t,
                                     size_t, void*);
template void pack_bf16<false, true>(const __bf16*, size_t, size_t, size_t,
                                    size_t, void*);
template void pack_bf16<true, false>(const __bf16*, size_t, size_t, size_t,
                                    size_t, void*);
template void pack_bf16<true, true>(const __bf16*, size_t, size_t, size_t,
                                   size_t, void*);

void pack_bf16(const __bf16* data, size_t rows, size_t cols,
              const PackingParams& params, void* out) {
  bool ti = params.transpose_inner;
  bool to = params.transpose_outer;
  if (ti && to)
    pack_bf16<true, true>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else if (ti)
    pack_bf16<true, false>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else if (to)
    pack_bf16<false, true>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else
    pack_bf16<false, false>(data, rows, cols, params.tile_rows, params.tile_cols, out);
}

// ---------- u8 ---------------------------------------------------------------

size_t packed_size_bytes_u8(size_t rows, size_t cols, size_t tile_r,
                            size_t tile_c) {
  return detail::packed_size_bytes<uint8_t>(rows, cols, tile_r, tile_c);
}

size_t packed_size_bytes_u8(size_t rows, size_t cols,
                            const PackingParams& params) {
  return packed_size_bytes_u8(rows, cols, params.tile_rows, params.tile_cols);
}

template <bool TransposeInner, bool TransposeOuter>
void pack_u8(const uint8_t* data, size_t rows, size_t cols, size_t tile_r,
             size_t tile_c, void* out) {
  detail::pack<uint8_t, TransposeInner, TransposeOuter>(data, rows, cols,
                                                         tile_r, tile_c, out);
}

template void pack_u8<false, false>(const uint8_t*, size_t, size_t, size_t,
                                    size_t, void*);
template void pack_u8<false, true>(const uint8_t*, size_t, size_t, size_t,
                                   size_t, void*);
template void pack_u8<true, false>(const uint8_t*, size_t, size_t, size_t,
                                   size_t, void*);
template void pack_u8<true, true>(const uint8_t*, size_t, size_t, size_t,
                                  size_t, void*);

void pack_u8(const uint8_t* data, size_t rows, size_t cols,
             const PackingParams& params, void* out) {
  bool ti = params.transpose_inner;
  bool to = params.transpose_outer;
  if (ti && to)
    pack_u8<true, true>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else if (ti)
    pack_u8<true, false>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else if (to)
    pack_u8<false, true>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else
    pack_u8<false, false>(data, rows, cols, params.tile_rows, params.tile_cols, out);
}

// ---------- s8 ---------------------------------------------------------------

size_t packed_size_bytes_s8(size_t rows, size_t cols, size_t tile_r,
                            size_t tile_c) {
  return detail::packed_size_bytes<int8_t>(rows, cols, tile_r, tile_c);
}

size_t packed_size_bytes_s8(size_t rows, size_t cols,
                            const PackingParams& params) {
  return packed_size_bytes_s8(rows, cols, params.tile_rows, params.tile_cols);
}

template <bool TransposeInner, bool TransposeOuter>
void pack_s8(const int8_t* data, size_t rows, size_t cols, size_t tile_r,
             size_t tile_c, void* out) {
  detail::pack<int8_t, TransposeInner, TransposeOuter>(data, rows, cols,
                                                        tile_r, tile_c, out);
}

template void pack_s8<false, false>(const int8_t*, size_t, size_t, size_t,
                                    size_t, void*);
template void pack_s8<false, true>(const int8_t*, size_t, size_t, size_t,
                                   size_t, void*);
template void pack_s8<true, false>(const int8_t*, size_t, size_t, size_t,
                                   size_t, void*);
template void pack_s8<true, true>(const int8_t*, size_t, size_t, size_t,
                                  size_t, void*);

void pack_s8(const int8_t* data, size_t rows, size_t cols,
             const PackingParams& params, void* out) {
  bool ti = params.transpose_inner;
  bool to = params.transpose_outer;
  if (ti && to)
    pack_s8<true, true>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else if (ti)
    pack_s8<true, false>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else if (to)
    pack_s8<false, true>(data, rows, cols, params.tile_rows, params.tile_cols, out);
  else
    pack_s8<false, false>(data, rows, cols, params.tile_rows, params.tile_cols, out);
}

void compute_ksums_s8(const int8_t* data, size_t K, size_t N, const float* w_scales,
    float* out) {
  // Precompute the common per-channel term dependent on weights and weight scales.
  // Out[...] = (act - zp_act) * s_act * w * s_w ... (along k dim)
  // Compute sum of w * s_w along each column. This gives the per-channel scale
  // factor to apply activation zp in the kernel.

  // Data is row-major, KxN
  // We want to iterate along columns
  for (auto n = 0u; n < N; n++) {
    int64_t k_sum = 0;
    for (auto k = 0u; k < K; k++) {
      k_sum += data[k * N + n];
    }
    out[n] = static_cast<float>(k_sum) * w_scales[n];
  }
}

}  // namespace sme
