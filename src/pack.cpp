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

void compute_group_ksums_s8(const int8_t* data, size_t K, size_t N,
    size_t group_size, const float* w_scales, float* out) {
  // Precompute the accumulated per-channel ksum across all groups:
  //   out[n] = sum_g( w_scales[g*N+n] * sum_{k in group g}(data[k*N+n]) )
  size_t num_groups = (K + group_size - 1) / group_size;
  for (size_t n = 0; n < N; n++) {
    float total = 0.0f;
    for (size_t g = 0; g < num_groups; g++) {
      size_t k_start = g * group_size;
      size_t k_end = k_start + group_size;
      if (k_end > K) k_end = K;
      int64_t group_sum = 0;
      for (size_t k = k_start; k < k_end; k++) {
        group_sum += data[k * N + n];
      }
      total += static_cast<float>(group_sum) * w_scales[g * N + n];
    }
    out[n] = total;
  }
}

// ---------- group scale tiling ------------------------------------------------

size_t packed_group_scales_len(size_t num_groups, size_t N, size_t tile_n) {
  size_t n_tiles = (N + tile_n - 1) / tile_n;
  return n_tiles * num_groups * tile_n;
}

void pack_group_scales(const float* scales, size_t num_groups, size_t N,
                       size_t tile_n, float* out) {
  size_t n_tiles = (N + tile_n - 1) / tile_n;
  for (size_t nt = 0; nt < n_tiles; nt++) {
    for (size_t g = 0; g < num_groups; g++) {
      for (size_t no = 0; no < tile_n; no++) {
        size_t n = nt * tile_n + no;
        *out++ = (n < N) ? scales[g * N + n] : 0.0f;
      }
    }
  }
}

// ---------- s4 (signed 4-bit, nibble-packed) ---------------------------------
//
// Packs signed 4-bit values into nibble pairs, tiled identically to the s8
// packing but with two adjacent row-tiles folded into each byte:
//   lower nibble = element from row-tile 2k  (sign-magnitude in 4 bits)
//   upper nibble = element from row-tile 2k+1
//
// For the qc4w RHS (TransposeOuter=true), row-tiles correspond to K-tiles,
// so each output byte holds values from two consecutive K-tiles.

size_t packed_size_bytes_s4(size_t rows, size_t cols, size_t tile_r,
                            size_t tile_c) {
  size_t row_tiles = (rows + tile_r - 1) / tile_r;
  size_t col_tiles = (cols + tile_c - 1) / tile_c;
  // Pair row-tiles: each pair produces tile_r * tile_c bytes (one byte per
  // element position, containing two nibbles).
  size_t row_tile_pairs = (row_tiles + 1) / 2;
  return row_tile_pairs * col_tiles * tile_r * tile_c;
}

size_t packed_size_bytes_s4(size_t rows, size_t cols,
                            const PackingParams& params) {
  return packed_size_bytes_s4(rows, cols, params.tile_rows, params.tile_cols);
}

void pack_s4(const int8_t* data, size_t rows, size_t cols,
             const PackingParams& params, void* out) {
  size_t tile_r = params.tile_rows;
  size_t tile_c = params.tile_cols;

  size_t num_row_tiles = (rows + tile_r - 1) / tile_r;
  size_t num_col_tiles = (cols + tile_c - 1) / tile_c;

  // After TransposeOuter swap (same logic as detail::pack).
  size_t out_row_tiles = num_row_tiles;
  size_t out_col_tiles = num_col_tiles;
  if (params.transpose_outer) {
    std::swap(out_row_tiles, out_col_tiles);
  }

  // After TransposeInner swap.
  size_t out_tile_r = tile_r;
  size_t out_tile_c = tile_c;
  if (params.transpose_inner) {
    std::swap(out_tile_r, out_tile_c);
  }

  // We pair along the inner (c_tile) dimension of the output, which maps to
  // row-tiles after TransposeOuter.
  size_t out_col_tile_pairs = (out_col_tiles + 1) / 2;

  auto* dst = static_cast<uint8_t*>(out);

  for (size_t r_tile = 0; r_tile < out_row_tiles; r_tile++) {
    for (size_t c_pair = 0; c_pair < out_col_tile_pairs; c_pair++) {
      size_t c_tile_lo = c_pair * 2;
      size_t c_tile_hi = c_pair * 2 + 1;

      for (size_t r_off = 0; r_off < out_tile_r; r_off++) {
        for (size_t c_off = 0; c_off < out_tile_c; c_off++) {
          size_t in_r_off = params.transpose_inner ? c_off : r_off;
          size_t in_c_off = params.transpose_inner ? r_off : c_off;

          // Lower nibble: from c_tile_lo.
          size_t row0, col0;
          if (params.transpose_outer) {
            row0 = c_tile_lo * tile_r + in_r_off;
            col0 = r_tile * tile_c + in_c_off;
          } else {
            row0 = r_tile * tile_r + in_r_off;
            col0 = c_tile_lo * tile_c + in_c_off;
          }
          int8_t val0 = 0;
          if (row0 < rows && col0 < cols)
            val0 = data[row0 * cols + col0];

          // Upper nibble: from c_tile_hi.
          int8_t val1 = 0;
          if (c_tile_hi < out_col_tiles) {
            size_t row1, col1;
            if (params.transpose_outer) {
              row1 = c_tile_hi * tile_r + in_r_off;
              col1 = r_tile * tile_c + in_c_off;
            } else {
              row1 = r_tile * tile_r + in_r_off;
              col1 = c_tile_hi * tile_c + in_c_off;
            }
            if (row1 < rows && col1 < cols)
              val1 = data[row1 * cols + col1];
          }

          *dst++ = (static_cast<uint8_t>(val0) & 0xF) |
                   ((static_cast<uint8_t>(val1) & 0xF) << 4);
        }
      }
    }
  }
}

// ---------- deinterleaved packing for FMOPA rank-2 ---------------------------
//
// Within each subtile (tile_rows/2 rows × tile_cols=4 cols), the standard
// layout stores [r0_k0, r0_k1, r0_k2, r0_k3, r1_k0, ...].  The deinterleaved
// layout stores k01 for all rows first, then k23:
//   [r0_k0, r0_k1, r1_k0, r1_k1, ..., r0_k2, r0_k3, r1_k2, r1_k3, ...]
// After svunpklo/svunpkhi the two K-pairs land in separate f16 vectors
// directly usable by rank-2 FMOPA without a UZP step.

void pack_s8_deinterleaved(const int8_t* data, size_t rows, size_t cols,
                           const PackingParams& params, void* out) {
  size_t tile_r = params.tile_rows;
  size_t tile_c = params.tile_cols;  // must be 4
  size_t sub_r = tile_r / 2;

  size_t out_row_tiles = (rows + tile_r - 1) / tile_r;
  size_t out_col_tiles = (cols + tile_c - 1) / tile_c;

  auto* dst = static_cast<int8_t*>(out);

  for (size_t rt = 0; rt < out_row_tiles; rt++) {
    for (size_t ct = 0; ct < out_col_tiles; ct++) {
      for (size_t sub = 0; sub < 2; sub++) {
        // First half: k0, k1 for all rows in this subtile.
        for (size_t r = 0; r < sub_r; r++) {
          size_t row = rt * tile_r + sub * sub_r + r;
          for (size_t k = 0; k < 2; k++) {
            size_t col = ct * tile_c + k;
            *dst++ = (row < rows && col < cols) ? data[row * cols + col] : 0;
          }
        }
        // Second half: k2, k3 for all rows in this subtile.
        for (size_t r = 0; r < sub_r; r++) {
          size_t row = rt * tile_r + sub * sub_r + r;
          for (size_t k = 2; k < 4; k++) {
            size_t col = ct * tile_c + k;
            *dst++ = (row < rows && col < cols) ? data[row * cols + col] : 0;
          }
        }
      }
    }
  }
}

void pack_s4_deinterleaved(const int8_t* data, size_t rows, size_t cols,
                           const PackingParams& params, void* out) {
  size_t tile_r = params.tile_rows;
  size_t tile_c = params.tile_cols;

  size_t num_row_tiles = (rows + tile_r - 1) / tile_r;
  size_t num_col_tiles = (cols + tile_c - 1) / tile_c;

  size_t out_row_tiles = num_row_tiles;
  size_t out_col_tiles = num_col_tiles;
  if (params.transpose_outer) std::swap(out_row_tiles, out_col_tiles);

  size_t out_tile_r = tile_r;
  size_t out_tile_c = tile_c;
  if (params.transpose_inner) std::swap(out_tile_r, out_tile_c);

  size_t out_col_tile_pairs = (out_col_tiles + 1) / 2;

  auto get_val = [&](size_t r_tile, size_t c_tile,
                     size_t r_off, size_t c_off) -> int8_t {
    size_t in_r_off = params.transpose_inner ? c_off : r_off;
    size_t in_c_off = params.transpose_inner ? r_off : c_off;
    size_t row, col;
    if (params.transpose_outer) {
      row = c_tile * tile_r + in_r_off;
      col = r_tile * tile_c + in_c_off;
    } else {
      row = r_tile * tile_r + in_r_off;
      col = c_tile * tile_c + in_c_off;
    }
    if (row < rows && col < cols) return data[row * cols + col];
    return 0;
  };

  auto* dst = static_cast<uint8_t*>(out);

  for (size_t r_tile = 0; r_tile < out_row_tiles; r_tile++) {
    for (size_t c_pair = 0; c_pair < out_col_tile_pairs; c_pair++) {
      size_t c_tile_lo = c_pair * 2;
      size_t c_tile_hi = c_pair * 2 + 1;

      // First half: k0, k1 (c_off 0..1) for all spatial positions.
      for (size_t r_off = 0; r_off < out_tile_r; r_off++) {
        for (size_t c_off = 0; c_off < 2; c_off++) {
          int8_t val0 = get_val(r_tile, c_tile_lo, r_off, c_off);
          int8_t val1 = (c_tile_hi < out_col_tiles)
                            ? get_val(r_tile, c_tile_hi, r_off, c_off)
                            : 0;
          *dst++ = (static_cast<uint8_t>(val0) & 0xF) |
                   ((static_cast<uint8_t>(val1) & 0xF) << 4);
        }
      }

      // Second half: k2, k3 (c_off 2..3) for all spatial positions.
      for (size_t r_off = 0; r_off < out_tile_r; r_off++) {
        for (size_t c_off = 2; c_off < out_tile_c; c_off++) {
          int8_t val0 = get_val(r_tile, c_tile_lo, r_off, c_off);
          int8_t val1 = (c_tile_hi < out_col_tiles)
                            ? get_val(r_tile, c_tile_hi, r_off, c_off)
                            : 0;
          *dst++ = (static_cast<uint8_t>(val0) & 0xF) |
                   ((static_cast<uint8_t>(val1) & 0xF) << 4);
        }
      }
    }
  }
}

}  // namespace sme
