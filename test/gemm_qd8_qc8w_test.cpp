#include "gemm.h"
#include "pack.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace {

void fill_random_s8(int8_t* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-128, 127);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<int8_t>(dist(rng));
}

void fill_random_f32(float* buf, size_t n, unsigned seed,
                     float lo = 0.01f, float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

struct GemmShape {
  size_t M, N, K;
};

class GemmQd8Qc8wTest : public ::testing::TestWithParam<GemmShape> {};

// Validates the full pipeline: pack → kernel → compare to reference.
// DISABLED until the SME kernel is implemented (currently a zero-fill stub).
TEST_P(GemmQd8Qc8wTest, MatchesReference) {
  auto [M, N, K] = GetParam();

  auto pack = sme::gemm_qd8_qc8w_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(K, N, pack.rhs));
  std::vector<float> C(M * N, 0.0f);
  std::vector<float> C_ref(M * N, 0.0f);

  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s8(B.data(), B.size(), 123);

  std::vector<float> w_scales(N);
  fill_random_f32(w_scales.data(), N, 77);

  std::vector<float> w_ksums(N);
  sme::compute_ksums_s8(B.data(), K, N, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::QuantParams qp{/*.a_zero_point=*/5, /*.a_scale=*/0.05f,
                       /*.w_scales=*/w_scales.data(),
                       /*.w_ksums=*/w_ksums.data()};

  sme::pack_s8(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s8(B.data(), K, N, pack.rhs, rhs_packed.get());
  sme::gemm_qd8p_qc8wp_f32(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);
  sme::gemm_qd8_qc8w_f32_reference(p, A.data(), B.data(), C_ref.data(), qp);

  for (size_t i = 0; i < M * N; ++i) {
    float tol = 1e-4f + 1e-5f * std::fabs(C_ref[i]);
    ASSERT_NEAR(C[i], C_ref[i], tol)
        << "mismatch at flat index " << i;
  }
}

// Validates that the reference implementation, packing, and test harness
// are wired up correctly (does not exercise the SME kernel).
TEST_P(GemmQd8Qc8wTest, ReferencePipeline) {
  auto [M, N, K] = GetParam();

  auto pack = sme::gemm_qd8_qc8w_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);

  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s8(B.data(), B.size(), 123);

  std::vector<float> w_scales(N);
  fill_random_f32(w_scales.data(), N, 77);

  std::vector<float> w_ksums(N);
  sme::compute_ksums_s8(B.data(), K, N, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::QuantParams qp{5, 0.05f, w_scales.data(), w_ksums.data()};

  // Run reference on unpacked data.
  std::vector<float> C_ref(M * N, 0.0f);
  sme::gemm_qd8_qc8w_f32_reference(p, A.data(), B.data(), C_ref.data(), qp);

  // Check output is non-trivial (not all zeros).
  float sum = 0.0f;
  for (size_t i = 0; i < M * N; ++i) sum += std::fabs(C_ref[i]);
  EXPECT_GT(sum, 0.0f) << "reference produced all-zero output";

  // Verify packing round-trips don't crash and sizes are sane.
  size_t lhs_bytes = sme::packed_size_bytes_s8(M, K, pack.lhs);
  size_t rhs_bytes = sme::packed_size_bytes_s8(K, N, pack.rhs);
  EXPECT_GE(lhs_bytes, M * K * sizeof(int8_t));
  EXPECT_GE(rhs_bytes, K * N * sizeof(int8_t));

  auto lhs_packed = std::make_unique<char[]>(lhs_bytes);
  auto rhs_packed = std::make_unique<char[]>(rhs_bytes);
  sme::pack_s8(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_s8(B.data(), K, N, pack.rhs, rhs_packed.get());
}

INSTANTIATE_TEST_SUITE_P(
    EpilogueSingleTile, GemmQd8Qc8wTest,
    ::testing::Values(
        GemmShape{16, 16, 4},
        GemmShape{16, 16, 8},
        GemmShape{16, 16, 16}
    ),
    [](const auto& info) {
      auto s = info.param;
      return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
             std::to_string(s.N);
    });

INSTANTIATE_TEST_SUITE_P(
    EpilogueStrides, GemmQd8Qc8wTest,
    ::testing::Values(
        GemmShape{32, 16, 4},   // 2x1: 2 M-subtiles, 1 N-tile
        GemmShape{16, 32, 4},   // 1x2: 1 M-subtile, 2 N-tiles
        GemmShape{32, 32, 4},   // 2x2: 2 M-subtiles, 2 N-tiles
        GemmShape{48, 16, 4},   // 3x1: 3 M-subtiles, 1 N-tile
        GemmShape{32, 16, 8},   // 2x1 with 2 K-steps
        GemmShape{32, 32, 8}    // 2x2 with 2 K-steps
    ),
    [](const auto& info) {
      auto s = info.param;
      return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
             std::to_string(s.N);
    });

INSTANTIATE_TEST_SUITE_P(
    MainBody, GemmQd8Qc8wTest,
    ::testing::Values(
        GemmShape{64, 16, 4},    // 1 main body iter, 1 N-tile
        GemmShape{64, 32, 4},    // 1 main body iter, 2 N-tiles
        GemmShape{64, 16, 16},   // 1 main body iter, 4 K-steps
        GemmShape{128, 16, 4},   // 2 main body iters
        GemmShape{128, 32, 8},   // 2 main body iters, 2 N-tiles, 2 K-steps
        GemmShape{80, 16, 4},    // 1 main body + 1 epilogue subtile
        GemmShape{96, 32, 8},    // 1 main body + 2 epilogue subtiles, 2 N-tiles
        GemmShape{128, 128, 16}  // large square
    ),
    [](const auto& info) {
      auto s = info.param;
      return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
             std::to_string(s.N);
    });

INSTANTIATE_TEST_SUITE_P(
    PartialTiles, GemmQd8Qc8wTest,
    ::testing::Values(
        // Single axis partial
        GemmShape{17, 16, 4},    // partial M only
        GemmShape{16, 17, 4},    // partial N only
        GemmShape{16, 16, 5},    // partial K only
        // Multiple axis partial, epilogue only
        GemmShape{17, 17, 5},    // all partial, no main body
        GemmShape{33, 17, 7},    // 2 epilogue subtiles + partial M row, partial N, partial K
        // Main body + partial
        GemmShape{65, 16, 4},    // 1 main body + 1 partial M row
        GemmShape{64, 17, 4},    // 1 main body, partial N
        GemmShape{64, 16, 7},    // 1 main body, partial K
        GemmShape{65, 17, 7},    // 1 main body + partial on all axes
        GemmShape{80, 33, 12},   // 1 main body + 1 epilogue subtile, partial N, partial K
        GemmShape{129, 33, 9}    // 2 main body + partial M, partial N, partial K
    ),
    [](const auto& info) {
      auto s = info.param;
      return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
             std::to_string(s.N);
    });

}  // namespace
