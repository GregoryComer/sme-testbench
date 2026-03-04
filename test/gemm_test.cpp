#include "gemm.h"
#include "pack.h"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace {

// Change to true for deterministic sequential data.
constexpr bool kUseSequentialData = true;

void fill_random(float* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; ++i) buf[i] = dist(rng);
}

void fill_sequential(float* buf, size_t n, float start = 1.0f) {
  for (size_t i = 0; i < n; ++i) buf[i] = start + static_cast<float>(i);
}

// Element-wise comparison with a relative + absolute tolerance.
void expect_near(const float* actual, const float* expected, size_t n,
                 size_t K) {
  // Allow tolerance to grow with K (accumulation length).
  const float atol = 1e-5f * static_cast<float>(K);
  for (size_t i = 0; i < n; ++i) {
    float diff = std::fabs(actual[i] - expected[i]);
    ASSERT_LE(diff, atol)
        << "mismatch at flat index " << i << ": got " << actual[i]
        << " vs expected " << expected[i];
  }
}

struct GemmShape {
  size_t M, N, K;
};

class GemmF32Test : public ::testing::TestWithParam<GemmShape> {};

TEST_P(GemmF32Test, MatchesReference) {
  auto [M, N, K] = GetParam();

  auto pack = sme::gemm_f32_packing_params();

  std::vector<float> A(M * K), B(K * N);
  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f32(M, K, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_f32(K, N, pack.rhs));
  std::vector<float> C(M * N, 0.0f), C_ref(M * N, 0.0f);

  if constexpr (kUseSequentialData) {
    fill_sequential(A.data(), A.size(), 1.0f);
    fill_sequential(B.data(), B.size(), 1.0f);
  } else {
    fill_random(A.data(), A.size(), /*seed=*/42);
    fill_random(B.data(), B.size(), /*seed=*/123);
  }

  sme::GemmParams p{M, N, K};

  sme::pack_f32(A.data(), M, K, pack.lhs, lhs_packed.get());
  sme::pack_f32(B.data(), K, N, pack.rhs, rhs_packed.get());
  sme::gemm_f32p_f32p_f32(p, lhs_packed.get(), rhs_packed.get(), C.data());
  sme::gemm_f32_f32_f32_reference(p, A.data(), B.data(), C_ref.data());

  expect_near(C.data(), C_ref.data(), M * N, K);
}

// Small shapes for fast CI + edge-case coverage.
INSTANTIATE_TEST_SUITE_P(
    Small, GemmF32Test,
    ::testing::Values(
        GemmShape{1, 1, 1},
        GemmShape{1, 16, 16},
        GemmShape{4, 4, 4},
        GemmShape{7, 13, 5},
        GemmShape{16, 16, 1},
        GemmShape{16, 16, 16},
        GemmShape{1, 1024, 1024},
        GemmShape{16, 1024, 1024},
        GemmShape{128, 128, 128},
        GemmShape{32, 16, 2},
        GemmShape{16, 32, 2},
        GemmShape{32, 32, 2},
        GemmShape{32, 13, 2}
    ),
    [](const auto& info) {
      auto s = info.param;
      return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
             std::to_string(s.N);
    });

}  // namespace
