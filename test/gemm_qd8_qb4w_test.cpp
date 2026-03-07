#include "gemm.h"
#include "pack.h"

#include <cmath>
#include <cstdint>
#include <cstring>
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

void fill_random_s4(int8_t* buf, size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-8, 7);
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

auto shape_name = [](const auto& info) {
  auto s = info.param;
  return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
         std::to_string(s.N);
};

constexpr size_t kGroupSize = 32;

// Helper: run a qb4w kernel and compare to reference.
template <typename GemmFn, typename PackingFn>
void run_qd8_qb4w_test(size_t M, size_t N, size_t K, PackingFn packing_fn, GemmFn gemm_fn) {
  auto pack = packing_fn();

  size_t num_groups = (K + kGroupSize - 1) / kGroupSize;
  size_t K_padded = num_groups * kGroupSize;

  // Original data (M×K, K×N) for reference.
  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s4(B.data(), B.size(), 123);

  // LHS padded to M×K_padded (zero-padded columns).
  std::vector<int8_t> A_padded(M * K_padded, 0);
  for (size_t m = 0; m < M; m++)
    std::memcpy(&A_padded[m * K_padded], &A[m * K], K);

  // RHS padded to K_padded×N (zero-padded rows). Stride is N, so first K*N
  // elements are identical to B.
  std::vector<int8_t> B_padded(K_padded * N, 0);
  std::memcpy(B_padded.data(), B.data(), K * N);

  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K_padded, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s4(K_padded, N, pack.rhs));
  std::vector<float> C(M * N, 0.0f);
  std::vector<float> C_ref(M * N, 0.0f);

  std::vector<float> w_scales(num_groups * N);
  fill_random_f32(w_scales.data(), w_scales.size(), 77);

  std::vector<float> w_ksums(N);
  sme::compute_group_ksums_s8(B.data(), K, N, kGroupSize, w_scales.data(), w_ksums.data());

  size_t svl_w = pack.rhs.tile_cols;

  // Tile-pack w_scales: [num_groups][N] -> [N/svl_w][num_groups][svl_w].
  std::vector<float> packed_scales(
      sme::packed_group_scales_len(num_groups, N, svl_w));
  sme::pack_group_scales(w_scales.data(), num_groups, N, svl_w,
                         packed_scales.data());

  sme::GemmParams p{M, N, K_padded};
  sme::BlockQuantParams qp{/*.a_zero_point=*/0, /*.a_scale=*/0.05f,
                            /*.group_size=*/kGroupSize,
                            /*.w_scales=*/packed_scales.data(),
                            /*.w_ksums=*/w_ksums.data()};

  std::vector<float> scratch(4 * svl_w * svl_w);

  sme::pack_s8(A_padded.data(), M, K_padded, pack.lhs, lhs_packed.get());
  sme::pack_s4(B_padded.data(), K_padded, N, pack.rhs, rhs_packed.get());
  gemm_fn(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp, scratch.data());

  // Reference uses unpacked row-major scales.
  sme::GemmParams p_ref{M, N, K};
  sme::BlockQuantParams qp_ref{0, 0.05f, kGroupSize, w_scales.data(), w_ksums.data()};
  sme::gemm_qd8_qb4w_f32_reference(p_ref, A.data(), B.data(), C_ref.data(), qp_ref);

  for (size_t i = 0; i < M * N; ++i) {
    float tol = 1e-4f + 1e-5f * std::fabs(C_ref[i]);
    ASSERT_NEAR(C[i], C_ref[i], tol)
        << "mismatch at flat index " << i;
  }
}

// --- 4vlxvl (4x1) -----------------------------------------------------------

class GemmQd8Qb4w_4vlxvlTest : public ::testing::TestWithParam<GemmShape> {};

TEST_P(GemmQd8Qb4w_4vlxvlTest, MatchesReference) {
  auto [M, N, K] = GetParam();
  run_qd8_qb4w_test(M, N, K, sme::gemm_qd8_qb4w_4vlxvl_packing_params,
                     sme::gemm_qd8p_qb4wp_f32_4vlxvl);
}

TEST_P(GemmQd8Qb4w_4vlxvlTest, ReferencePipeline) {
  auto [M, N, K] = GetParam();

  size_t num_groups = (K + kGroupSize - 1) / kGroupSize;
  size_t K_padded = num_groups * kGroupSize;
  auto pack = sme::gemm_qd8_qb4w_4vlxvl_packing_params();

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);

  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s4(B.data(), B.size(), 123);

  std::vector<float> w_scales(num_groups * N);
  fill_random_f32(w_scales.data(), w_scales.size(), 77);

  std::vector<float> w_ksums(N);
  sme::compute_group_ksums_s8(B.data(), K, N, kGroupSize, w_scales.data(), w_ksums.data());

  sme::GemmParams p{M, N, K};
  sme::BlockQuantParams qp{5, 0.05f, kGroupSize, w_scales.data(), w_ksums.data()};

  // Run reference on unpacked data.
  std::vector<float> C_ref(M * N, 0.0f);
  sme::gemm_qd8_qb4w_f32_reference(p, A.data(), B.data(), C_ref.data(), qp);

  // Check output is non-trivial (not all zeros).
  float sum = 0.0f;
  for (size_t i = 0; i < M * N; ++i) sum += std::fabs(C_ref[i]);
  EXPECT_GT(sum, 0.0f) << "reference produced all-zero output";

  // Verify packing round-trips don't crash and sizes are sane.
  // LHS padded to K_padded columns for packing.
  std::vector<int8_t> A_padded(M * K_padded, 0);
  for (size_t m = 0; m < M; m++)
    std::memcpy(&A_padded[m * K_padded], &A[m * K], K);

  size_t lhs_bytes = sme::packed_size_bytes_s8(M, K_padded, pack.lhs);
  size_t rhs_bytes = sme::packed_size_bytes_s4(K_padded, N, pack.rhs);
  EXPECT_GE(lhs_bytes, M * K * sizeof(int8_t));
  // s4 packed size should be at least half the element count (nibble-packed).
  EXPECT_GE(rhs_bytes, (K * N + 1) / 2);

  auto lhs_packed = std::make_unique<char[]>(lhs_bytes);
  sme::pack_s8(A_padded.data(), M, K_padded, pack.lhs, lhs_packed.get());
}

INSTANTIATE_TEST_SUITE_P(EpilogueSingleTile, GemmQd8Qb4w_4vlxvlTest,
    ::testing::Values(GemmShape{16, 16, 32}, GemmShape{16, 16, 64}, GemmShape{16, 16, 128}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(EpilogueStrides, GemmQd8Qb4w_4vlxvlTest,
    ::testing::Values(
        GemmShape{32, 16, 32}, GemmShape{16, 32, 32}, GemmShape{32, 32, 32},
        GemmShape{48, 16, 32}, GemmShape{32, 16, 64}, GemmShape{32, 32, 64}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(MainBody, GemmQd8Qb4w_4vlxvlTest,
    ::testing::Values(
        GemmShape{64, 16, 32}, GemmShape{64, 32, 32}, GemmShape{64, 16, 128},
        GemmShape{128, 16, 32}, GemmShape{128, 32, 64}, GemmShape{80, 16, 32},
        GemmShape{96, 32, 64}, GemmShape{128, 128, 128}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(PartialTiles, GemmQd8Qb4w_4vlxvlTest,
    ::testing::Values(
        GemmShape{17, 16, 32}, GemmShape{16, 17, 32}, GemmShape{16, 16, 64},
        GemmShape{17, 17, 32}, GemmShape{33, 17, 64}, GemmShape{65, 16, 32},
        GemmShape{64, 17, 32}, GemmShape{64, 16, 64}, GemmShape{65, 17, 64},
        GemmShape{80, 33, 96}, GemmShape{129, 33, 64}),
    shape_name);

// --- 2vlx2vl (2x2) ----------------------------------------------------------

class GemmQd8Qb4w_2vlx2vlTest : public ::testing::TestWithParam<GemmShape> {};

TEST_P(GemmQd8Qb4w_2vlx2vlTest, MatchesReference) {
  auto [M, N, K] = GetParam();
  run_qd8_qb4w_test(M, N, K, sme::gemm_qd8_qb4w_2vlx2vl_packing_params,
      [](const sme::GemmParams& p, const void* lhs, const void* rhs,
         float* out, const sme::BlockQuantParams& qp, float*) {
        sme::gemm_qd8p_qb4wp_f32_2vlx2vl(p, lhs, rhs, out, qp);
      });
}

INSTANTIATE_TEST_SUITE_P(EpilogueSingleTile, GemmQd8Qb4w_2vlx2vlTest,
    ::testing::Values(GemmShape{16, 16, 32}, GemmShape{16, 16, 64}, GemmShape{16, 16, 128}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(EpilogueStrides, GemmQd8Qb4w_2vlx2vlTest,
    ::testing::Values(
        GemmShape{32, 16, 32}, GemmShape{16, 32, 32}, GemmShape{32, 32, 32},
        GemmShape{48, 16, 32}, GemmShape{32, 16, 64}, GemmShape{32, 32, 64}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(MainBody, GemmQd8Qb4w_2vlx2vlTest,
    ::testing::Values(
        GemmShape{64, 16, 32}, GemmShape{64, 32, 32}, GemmShape{64, 16, 128},
        GemmShape{128, 16, 32}, GemmShape{128, 32, 64}, GemmShape{80, 16, 32},
        GemmShape{96, 32, 64}, GemmShape{128, 128, 128}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(PartialTiles, GemmQd8Qb4w_2vlx2vlTest,
    ::testing::Values(
        GemmShape{17, 16, 32}, GemmShape{16, 17, 32}, GemmShape{16, 16, 64},
        GemmShape{17, 17, 32}, GemmShape{33, 17, 64}, GemmShape{65, 16, 32},
        GemmShape{64, 17, 32}, GemmShape{64, 16, 64}, GemmShape{65, 17, 64},
        GemmShape{80, 33, 96}, GemmShape{129, 33, 64}),
    shape_name);

}  // namespace
