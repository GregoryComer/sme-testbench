#include "gemm.h"
#include "pack.h"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace {

void fill_random(_Float16* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<_Float16>(dist(rng));
}

void expect_near(const _Float16* actual, const _Float16* expected, size_t n,
                 size_t K) {
  // f16 has ~3 decimal digits of precision; tolerance scales with K.
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

auto shape_name = [](const auto& info) {
  auto s = info.param;
  return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
         std::to_string(s.N);
};

// --- 4vlxvl (4x1) -----------------------------------------------------------

class GemmF16_4vlxvlTest : public ::testing::TestWithParam<GemmShape> {};

TEST_P(GemmF16_4vlxvlTest, MatchesReference) {
  auto [M, N, K] = GetParam();

  auto pack = sme::gemm_f16_4vlxvl_packing_params();

  std::vector<_Float16> A(M * K), B(K * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f16(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f16(K, N, pack.rhs));
  std::vector<_Float16> C(M * N, static_cast<_Float16>(0));
  std::vector<_Float16> C_ref(M * N, static_cast<_Float16>(0));

  fill_random(A.data(), A.size(), 42);
  fill_random(B.data(), B.size(), 123);

  sme::GemmParams p{M, N, K};

  sme::pack_f16(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_f16(B.data(), K, N, pack.rhs, rhs_packed.get());
  sme::gemm_f16p_f16p_f16_4vlxvl(p, lhs_packed.get(), rhs_packed.get(), C.data());
  sme::gemm_f16_f16_f16_reference(p, A.data(), B.data(), C_ref.data());

  expect_near(C.data(), C_ref.data(), M * N, K);
}

INSTANTIATE_TEST_SUITE_P(EpilogueSingleTile, GemmF16_4vlxvlTest,
    ::testing::Values(GemmShape{16, 16, 2}, GemmShape{16, 16, 16}), shape_name);

INSTANTIATE_TEST_SUITE_P(EpilogueStrides, GemmF16_4vlxvlTest,
    ::testing::Values(
        GemmShape{32, 16, 2}, GemmShape{16, 32, 2}, GemmShape{32, 32, 2},
        GemmShape{48, 16, 2}, GemmShape{32, 16, 4}, GemmShape{32, 32, 4}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(MainBody, GemmF16_4vlxvlTest,
    ::testing::Values(
        GemmShape{64, 16, 2}, GemmShape{64, 32, 2}, GemmShape{64, 16, 16},
        GemmShape{128, 16, 2}, GemmShape{128, 32, 4}, GemmShape{80, 16, 2},
        GemmShape{96, 32, 4}, GemmShape{128, 128, 16}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(PartialTiles, GemmF16_4vlxvlTest,
    ::testing::Values(
        GemmShape{17, 16, 2}, GemmShape{16, 17, 2}, GemmShape{16, 16, 3},
        GemmShape{17, 17, 3}, GemmShape{33, 17, 5}, GemmShape{65, 16, 2},
        GemmShape{64, 17, 2}, GemmShape{64, 16, 5}, GemmShape{65, 17, 5},
        GemmShape{80, 33, 7}, GemmShape{129, 33, 7}),
    shape_name);

// --- 2vlx2vl (2x2) -----------------------------------------------------------

class GemmF16_2vlx2vlTest : public ::testing::TestWithParam<GemmShape> {};

TEST_P(GemmF16_2vlx2vlTest, MatchesReference) {
  auto [M, N, K] = GetParam();

  auto pack = sme::gemm_f16_2vlx2vl_packing_params();

  std::vector<_Float16> A(M * K), B(K * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f16(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f16(K, N, pack.rhs));
  std::vector<_Float16> C(M * N, static_cast<_Float16>(0));
  std::vector<_Float16> C_ref(M * N, static_cast<_Float16>(0));

  fill_random(A.data(), A.size(), 42);
  fill_random(B.data(), B.size(), 123);

  sme::GemmParams p{M, N, K};

  sme::pack_f16(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_f16(B.data(), K, N, pack.rhs, rhs_packed.get());
  sme::gemm_f16p_f16p_f16_2vlx2vl(p, lhs_packed.get(), rhs_packed.get(), C.data());
  sme::gemm_f16_f16_f16_reference(p, A.data(), B.data(), C_ref.data());

  expect_near(C.data(), C_ref.data(), M * N, K);
}

INSTANTIATE_TEST_SUITE_P(EpilogueSingleTile, GemmF16_2vlx2vlTest,
    ::testing::Values(GemmShape{16, 16, 2}, GemmShape{16, 16, 16}), shape_name);

INSTANTIATE_TEST_SUITE_P(EpilogueStrides, GemmF16_2vlx2vlTest,
    ::testing::Values(
        GemmShape{32, 16, 2}, GemmShape{16, 32, 2}, GemmShape{32, 32, 2},
        GemmShape{48, 16, 2}, GemmShape{32, 16, 4}, GemmShape{32, 32, 4}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(MainBody, GemmF16_2vlx2vlTest,
    ::testing::Values(
        GemmShape{64, 16, 2}, GemmShape{64, 32, 2}, GemmShape{64, 16, 16},
        GemmShape{128, 16, 2}, GemmShape{128, 32, 4}, GemmShape{80, 16, 2},
        GemmShape{96, 32, 4}, GemmShape{128, 128, 16}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(PartialTiles, GemmF16_2vlx2vlTest,
    ::testing::Values(
        GemmShape{17, 16, 2}, GemmShape{16, 17, 2}, GemmShape{16, 16, 3},
        GemmShape{17, 17, 3}, GemmShape{33, 17, 5}, GemmShape{65, 16, 2},
        GemmShape{64, 17, 2}, GemmShape{64, 16, 5}, GemmShape{65, 17, 5},
        GemmShape{80, 33, 7}, GemmShape{129, 33, 7}),
    shape_name);

}  // namespace
