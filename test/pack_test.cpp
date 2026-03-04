#include "pack.h"

#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include <gtest/gtest.h>

namespace {

// Fill with sequential values 1, 2, 3, ... for easy visual inspection.
std::vector<float> sequential(size_t n) {
  std::vector<float> v(n);
  std::iota(v.begin(), v.end(), 1.0f);
  return v;
}

// Read packed output as a flat float vector.
std::vector<float> to_vec(const void* buf, size_t count) {
  const auto* f = static_cast<const float*>(buf);
  return {f, f + count};
}

// Format a flat buffer as a 2D grid of tiles for readable failure output.
//
//   data      - flat float buffer
//   num_tiles - total number of tiles
//   tile_r    - rows per tile
//   tile_c    - cols per tile
//
// Output: one tile per block, separated by blank lines.
std::string format_tiled(const std::vector<float>& data, size_t num_tiles,
                         size_t tile_r, size_t tile_c) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(0);
  size_t idx = 0;
  for (size_t t = 0; t < num_tiles; t++) {
    os << "  tile " << t << ":\n";
    for (size_t r = 0; r < tile_r; r++) {
      os << "    ";
      for (size_t c = 0; c < tile_c; c++) {
        if (idx < data.size())
          os << std::setw(4) << data[idx];
        else
          os << "   ?";
        idx++;
      }
      os << "\n";
    }
  }
  return os.str();
}

// Format row-major input as a 2D grid.
std::string format_matrix(const std::vector<float>& data, size_t rows,
                          size_t cols) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(0);
  for (size_t r = 0; r < rows; r++) {
    os << "    ";
    for (size_t c = 0; c < cols; c++) {
      os << std::setw(4) << data[r * cols + c];
    }
    os << "\n";
  }
  return os.str();
}

// Build the expected packed output for a given transpose configuration, then
// compare against actual.  On mismatch, print input, expected, and actual as
// formatted grids.
template <bool TransposeInner, bool TransposeOuter>
void verify_pack(const std::vector<float>& input, size_t rows, size_t cols,
                 size_t tile_r, size_t tile_c, const void* packed_buf) {
  size_t r_tiles = (rows + tile_r - 1) / tile_r;
  size_t c_tiles = (cols + tile_c - 1) / tile_c;
  size_t num_tiles = r_tiles * c_tiles;
  size_t total = num_tiles * tile_r * tile_c;

  // Build expected.
  std::vector<float> expected;
  expected.reserve(total);

  auto emit_tile = [&](size_t rt, size_t ct) {
    if constexpr (TransposeInner) {
      // column-major within tile
      for (size_t c = 0; c < tile_c; c++) {
        for (size_t r = 0; r < tile_r; r++) {
          size_t sr = rt * tile_r + r;
          size_t sc = ct * tile_c + c;
          expected.push_back((sr < rows && sc < cols)
                                 ? input[sr * cols + sc]
                                 : 0.0f);
        }
      }
    } else {
      // row-major within tile
      for (size_t r = 0; r < tile_r; r++) {
        for (size_t c = 0; c < tile_c; c++) {
          size_t sr = rt * tile_r + r;
          size_t sc = ct * tile_c + c;
          expected.push_back((sr < rows && sc < cols)
                                 ? input[sr * cols + sc]
                                 : 0.0f);
        }
      }
    }
  };

  if constexpr (TransposeOuter) {
    for (size_t ct = 0; ct < c_tiles; ct++)
      for (size_t rt = 0; rt < r_tiles; rt++)
        emit_tile(rt, ct);
  } else {
    for (size_t rt = 0; rt < r_tiles; rt++)
      for (size_t ct = 0; ct < c_tiles; ct++)
        emit_tile(rt, ct);
  }

  auto actual = to_vec(packed_buf, total);

  // Element-wise compare with pretty failure.
  for (size_t i = 0; i < total; i++) {
    if (actual[i] != expected[i]) {
      // Determine which tile and offset.
      size_t elems_per_tile = tile_r * tile_c;
      size_t tile_idx = i / elems_per_tile;
      size_t in_tile = i % elems_per_tile;
      size_t off_r = in_tile / tile_c;
      size_t off_c = in_tile % tile_c;
      if constexpr (TransposeInner) {
        off_r = in_tile % tile_r;
        off_c = in_tile / tile_r;
      }

      FAIL() << "Mismatch at packed index " << i << " (tile " << tile_idx
             << ", offset " << off_r << "," << off_c << "): got " << actual[i]
             << ", expected " << expected[i]
             << "\n\nInput (" << rows << "x" << cols << "):\n"
             << format_matrix(input, rows, cols)
             << "\nExpected packed (" << num_tiles << " tiles of " << tile_r
             << "x" << tile_c << "):\n"
             << format_tiled(expected, num_tiles, tile_r, tile_c)
             << "\nActual packed:\n"
             << format_tiled(actual, num_tiles, tile_r, tile_c);
    }
  }
}

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------

struct PackParams {
  size_t rows, cols, tile_r, tile_c;
  bool transpose_inner;
  bool transpose_outer;
};

std::string name_pack_params(const ::testing::TestParamInfo<PackParams>& info) {
  auto& p = info.param;
  std::string s = std::to_string(p.rows) + "x" + std::to_string(p.cols) +
                  "_tile" + std::to_string(p.tile_r) + "x" +
                  std::to_string(p.tile_c);
  if (p.transpose_inner) s += "_TI";
  if (p.transpose_outer) s += "_TO";
  return s;
}

// ---------------------------------------------------------------------------
// TransposeInner=false, TransposeOuter=false  (plain tiled packing)
// ---------------------------------------------------------------------------

class PackF32_FF : public ::testing::TestWithParam<PackParams> {};

TEST_P(PackF32_FF, Pack) {
  auto [rows, cols, tile_r, tile_c, ti, to_] = GetParam();

  auto input = sequential(rows * cols);
  auto buf = std::make_unique<char[]>(sme::packed_size_bytes_f32(rows, cols, tile_r, tile_c));
  sme::pack_f32<false, false>(input.data(), rows, cols, tile_r, tile_c,
                               buf.get());

  verify_pack<false, false>(input, rows, cols, tile_r, tile_c, buf.get());
}

INSTANTIATE_TEST_SUITE_P(
    SingleTile, PackF32_FF,
    ::testing::Values(
        PackParams{4, 4, 4, 4, false, false},
        PackParams{8, 8, 8, 8, false, false},
        PackParams{1, 1, 1, 1, false, false}),
    name_pack_params);

INSTANTIATE_TEST_SUITE_P(
    PartialTile, PackF32_FF,
    ::testing::Values(
        PackParams{3, 3, 4, 4, false, false},
        PackParams{1, 7, 4, 8, false, false},
        PackParams{5, 1, 8, 4, false, false}),
    name_pack_params);

INSTANTIATE_TEST_SUITE_P(
    MultiTile, PackF32_FF,
    ::testing::Values(
        PackParams{8, 8, 4, 4, false, false},
        PackParams{16, 16, 4, 4, false, false},
        PackParams{7, 13, 4, 4, false, false},
        PackParams{9, 6, 4, 4, false, false}),
    name_pack_params);

// ---------------------------------------------------------------------------
// TransposeInner=true, TransposeOuter=false
// ---------------------------------------------------------------------------

class PackF32_TF : public ::testing::TestWithParam<PackParams> {};

TEST_P(PackF32_TF, Pack) {
  auto [rows, cols, tile_r, tile_c, ti, to_] = GetParam();

  auto input = sequential(rows * cols);
  auto buf = std::make_unique<char[]>(sme::packed_size_bytes_f32(rows, cols, tile_r, tile_c));
  sme::pack_f32<true, false>(input.data(), rows, cols, tile_r, tile_c,
                              buf.get());

  verify_pack<true, false>(input, rows, cols, tile_r, tile_c, buf.get());
}

INSTANTIATE_TEST_SUITE_P(
    SingleTile, PackF32_TF,
    ::testing::Values(
        PackParams{4, 4, 4, 4, true, false},
        PackParams{1, 1, 1, 1, true, false}),
    name_pack_params);

INSTANTIATE_TEST_SUITE_P(
    PartialTile, PackF32_TF,
    ::testing::Values(
        PackParams{3, 3, 4, 4, true, false},
        PackParams{1, 7, 4, 8, true, false}),
    name_pack_params);

INSTANTIATE_TEST_SUITE_P(
    MultiTile, PackF32_TF,
    ::testing::Values(
        PackParams{8, 8, 4, 4, true, false},
        PackParams{7, 13, 4, 4, true, false}),
    name_pack_params);

// ---------------------------------------------------------------------------
// TransposeInner=false, TransposeOuter=true
// ---------------------------------------------------------------------------

class PackF32_FT : public ::testing::TestWithParam<PackParams> {};

TEST_P(PackF32_FT, Pack) {
  auto [rows, cols, tile_r, tile_c, ti, to_] = GetParam();

  auto input = sequential(rows * cols);
  auto buf = std::make_unique<char[]>(sme::packed_size_bytes_f32(rows, cols, tile_r, tile_c));
  sme::pack_f32<false, true>(input.data(), rows, cols, tile_r, tile_c,
                              buf.get());

  verify_pack<false, true>(input, rows, cols, tile_r, tile_c, buf.get());
}

INSTANTIATE_TEST_SUITE_P(
    SingleTile, PackF32_FT,
    ::testing::Values(
        PackParams{4, 4, 4, 4, false, true},
        PackParams{1, 1, 1, 1, false, true}),
    name_pack_params);

INSTANTIATE_TEST_SUITE_P(
    PartialTile, PackF32_FT,
    ::testing::Values(
        PackParams{3, 3, 4, 4, false, true},
        PackParams{5, 1, 8, 4, false, true}),
    name_pack_params);

INSTANTIATE_TEST_SUITE_P(
    MultiTile, PackF32_FT,
    ::testing::Values(
        PackParams{8, 8, 4, 4, false, true},
        PackParams{7, 13, 4, 4, false, true}),
    name_pack_params);

// ---------------------------------------------------------------------------
// TransposeInner=true, TransposeOuter=true
// ---------------------------------------------------------------------------

class PackF32_TT : public ::testing::TestWithParam<PackParams> {};

TEST_P(PackF32_TT, Pack) {
  auto [rows, cols, tile_r, tile_c, ti, to_] = GetParam();

  auto input = sequential(rows * cols);
  auto buf = std::make_unique<char[]>(sme::packed_size_bytes_f32(rows, cols, tile_r, tile_c));
  sme::pack_f32<true, true>(input.data(), rows, cols, tile_r, tile_c,
                             buf.get());

  verify_pack<true, true>(input, rows, cols, tile_r, tile_c, buf.get());
}

INSTANTIATE_TEST_SUITE_P(
    SingleTile, PackF32_TT,
    ::testing::Values(
        PackParams{4, 4, 4, 4, true, true},
        PackParams{1, 1, 1, 1, true, true}),
    name_pack_params);

INSTANTIATE_TEST_SUITE_P(
    PartialTile, PackF32_TT,
    ::testing::Values(
        PackParams{3, 3, 4, 4, true, true},
        PackParams{1, 7, 4, 8, true, true}),
    name_pack_params);

INSTANTIATE_TEST_SUITE_P(
    MultiTile, PackF32_TT,
    ::testing::Values(
        PackParams{8, 8, 4, 4, true, true},
        PackParams{7, 13, 4, 4, true, true}),
    name_pack_params);

// ---------------------------------------------------------------------------
// Hardcoded golden tests — one per transpose permutation.
//
// Input: 3x5, tile 2x3 → 2 row-tiles x 2 col-tiles, partial on both axes.
//
//   Input (3x5):
//     1  2  3  4  5
//     6  7  8  9 10
//    11 12 13 14 15
//
// ---------------------------------------------------------------------------

// FF: tiles row-major, elements row-major within tile.
//   tile(0,0): rows[0..1] cols[0..2]   tile(0,1): rows[0..1] cols[3..4+pad]
//   tile(1,0): rows[2..2+pad] cols[0..2] tile(1,1): rows[2..2+pad] cols[3..4+pad]
TEST(PackF32Golden, FF_3x5_tile2x3) {
  // clang-format off
  std::vector<float> input = {
     1,  2,  3,  4,  5,
     6,  7,  8,  9, 10,
    11, 12, 13, 14, 15,
  };
  std::vector<float> expected = {
    // tile(0,0)
     1,  2,  3,
     6,  7,  8,
    // tile(0,1)
     4,  5,  0,
     9, 10,  0,
    // tile(1,0)
    11, 12, 13,
     0,  0,  0,
    // tile(1,1)
    14, 15,  0,
     0,  0,  0,
  };
  // clang-format on
  auto buf = std::make_unique<char[]>(sme::packed_size_bytes_f32(3, 5, 2, 3));
  sme::pack_f32<false, false>(input.data(), 3, 5, 2, 3, buf.get());
  auto actual = to_vec(buf.get(), expected.size());
  EXPECT_EQ(actual, expected);
}

// TF: tiles row-major, elements column-major within tile.
TEST(PackF32Golden, TF_3x5_tile2x3) {
  // clang-format off
  std::vector<float> input = {
     1,  2,  3,  4,  5,
     6,  7,  8,  9, 10,
    11, 12, 13, 14, 15,
  };
  std::vector<float> expected = {
    // tile(0,0): cols then rows
     1,  6,
     2,  7,
     3,  8,
    // tile(0,1)
     4,  9,
     5, 10,
     0,  0,
    // tile(1,0)
    11,  0,
    12,  0,
    13,  0,
    // tile(1,1)
    14,  0,
    15,  0,
     0,  0,
  };
  // clang-format on
  auto buf = std::make_unique<char[]>(sme::packed_size_bytes_f32(3, 5, 2, 3));
  sme::pack_f32<true, false>(input.data(), 3, 5, 2, 3, buf.get());
  auto actual = to_vec(buf.get(), expected.size());
  EXPECT_EQ(actual, expected);
}

// FT: tiles column-major, elements row-major within tile.
TEST(PackF32Golden, FT_3x5_tile2x3) {
  // clang-format off
  std::vector<float> input = {
     1,  2,  3,  4,  5,
     6,  7,  8,  9, 10,
    11, 12, 13, 14, 15,
  };
  std::vector<float> expected = {
    // tile(0,0)
     1,  2,  3,
     6,  7,  8,
    // tile(1,0)
    11, 12, 13,
     0,  0,  0,
    // tile(0,1)
     4,  5,  0,
     9, 10,  0,
    // tile(1,1)
    14, 15,  0,
     0,  0,  0,
  };
  // clang-format on
  auto buf = std::make_unique<char[]>(sme::packed_size_bytes_f32(3, 5, 2, 3));
  sme::pack_f32<false, true>(input.data(), 3, 5, 2, 3, buf.get());
  auto actual = to_vec(buf.get(), expected.size());
  EXPECT_EQ(actual, expected);
}

// TT: tiles column-major, elements column-major within tile.
TEST(PackF32Golden, TT_3x5_tile2x3) {
  // clang-format off
  std::vector<float> input = {
     1,  2,  3,  4,  5,
     6,  7,  8,  9, 10,
    11, 12, 13, 14, 15,
  };
  std::vector<float> expected = {
    // tile(0,0): cols then rows
     1,  6,
     2,  7,
     3,  8,
    // tile(1,0)
    11,  0,
    12,  0,
    13,  0,
    // tile(0,1)
     4,  9,
     5, 10,
     0,  0,
    // tile(1,1)
    14,  0,
    15,  0,
     0,  0,
  };
  // clang-format on
  auto buf = std::make_unique<char[]>(sme::packed_size_bytes_f32(3, 5, 2, 3));
  sme::pack_f32<true, true>(input.data(), 3, 5, 2, 3, buf.get());
  auto actual = to_vec(buf.get(), expected.size());
  EXPECT_EQ(actual, expected);
}

}  // namespace
