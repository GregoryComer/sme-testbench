#include "gemm.h"
#include "pack.h"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace {

void fill_random(__bf16* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<__bf16>(dist(rng));
}

void expect_near(const __bf16* actual, const __bf16* expected, size_t n,
                 size_t K) {
  // bf16 has ~3 decimal digits of precision; tolerance scales with K.
  const float atol = 1e-2f * static_cast<float>(K);
  for (size_t i = 0; i < n; ++i) {
    float diff = std::fabs(static_cast<float>(actual[i]) -
                           static_cast<float>(expected[i]));
    ASSERT_LE(diff, atol)
        << "mismatch at flat index " << i << ": got "
        << static_cast<float>(actual[i]) << " vs expected "
        << static_cast<float>(expected[i]);
  }
}

struct GemmShape {
  size_t M, N, K;
};

class GemmBF16Test : public ::testing::TestWithParam<GemmShape> {};

TEST_P(GemmBF16Test, MatchesReference) {
  auto [M, N, K] = GetParam();

  auto pack = sme::gemm_bf16_packing_params();

  std::vector<__bf16> A(M * K), B(K * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_bf16(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_bf16(K, N, pack.rhs));
  std::vector<__bf16> C(M * N, static_cast<__bf16>(0));
  std::vector<__bf16> C_ref(M * N, static_cast<__bf16>(0));

  fill_random(A.data(), A.size(), 42);
  fill_random(B.data(), B.size(), 123);

  sme::GemmParams p{M, N, K};

  sme::pack_bf16(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_bf16(B.data(), K, N, pack.rhs, rhs_packed.get());
  sme::gemm_bf16p_bf16p_bf16(p, lhs_packed.get(), rhs_packed.get(), C.data());
  sme::gemm_bf16_bf16_bf16_reference(p, A.data(), B.data(), C_ref.data());

  expect_near(C.data(), C_ref.data(), M * N, K);
}

INSTANTIATE_TEST_SUITE_P(
    EpilogueSingleTile, GemmBF16Test,
    ::testing::Values(
        GemmShape{16, 16, 2},
        GemmShape{16, 16, 16}
    ),
    [](const auto& info) {
      auto s = info.param;
      return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
             std::to_string(s.N);
    });

INSTANTIATE_TEST_SUITE_P(
    EpilogueStrides, GemmBF16Test,
    ::testing::Values(
        GemmShape{32, 16, 2},   // 2x1: 2 M-subtiles, 1 N-tile
        GemmShape{16, 32, 2},   // 1x2: 1 M-subtile, 2 N-tiles
        GemmShape{32, 32, 2},   // 2x2: 2 M-subtiles, 2 N-tiles
        GemmShape{48, 16, 2},   // 3x1: 3 M-subtiles, 1 N-tile
        GemmShape{32, 16, 4},   // 2x1 with 2 K-steps
        GemmShape{32, 32, 4}    // 2x2 with 2 K-steps
    ),
    [](const auto& info) {
      auto s = info.param;
      return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
             std::to_string(s.N);
    });

INSTANTIATE_TEST_SUITE_P(
    MainBody, GemmBF16Test,
    ::testing::Values(
        GemmShape{64, 16, 2},    // 1 main body iter, 1 N-tile
        GemmShape{64, 32, 2},    // 1 main body iter, 2 N-tiles
        GemmShape{64, 16, 16},   // 1 main body iter, 8 K-steps
        GemmShape{128, 16, 2},   // 2 main body iters
        GemmShape{128, 32, 4},   // 2 main body iters, 2 N-tiles, 2 K-steps
        GemmShape{80, 16, 2},    // 1 main body + 1 epilogue subtile
        GemmShape{96, 32, 4},    // 1 main body + 2 epilogue subtiles, 2 N-tiles
        GemmShape{128, 128, 16}  // large square
    ),
    [](const auto& info) {
      auto s = info.param;
      return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
             std::to_string(s.N);
    });

INSTANTIATE_TEST_SUITE_P(
    PartialTiles, GemmBF16Test,
    ::testing::Values(
        // Single axis partial
        GemmShape{17, 16, 2},    // partial M only
        GemmShape{16, 17, 2},    // partial N only
        GemmShape{16, 16, 3},    // partial K only
        // Multiple axis partial, epilogue only
        GemmShape{17, 17, 3},    // all partial, no main body
        GemmShape{33, 17, 5},    // 2 epilogue subtiles + partial M row, partial N, partial K
        // Main body + partial
        GemmShape{65, 16, 2},    // 1 main body + 1 partial M row
        GemmShape{64, 17, 2},    // 1 main body, partial N
        GemmShape{64, 16, 5},    // 1 main body, partial K
        GemmShape{65, 17, 5},    // 1 main body + partial on all axes
        GemmShape{80, 33, 7},    // 1 main body + 1 epilogue subtile, partial N, partial K
        GemmShape{129, 33, 7}    // 2 main body + partial M, partial N, partial K
    ),
    [](const auto& info) {
      auto s = info.param;
      return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
             std::to_string(s.N);
    });

}  // namespace
