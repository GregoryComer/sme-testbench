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

// Overload with custom tolerance for f16 intermediate precision kernels.
// Optional pack function pointers allow using deinterleaved packing.
using PackS8Fn = void(*)(const int8_t*, size_t, size_t, const sme::PackingParams&, void*);
using PackS4Fn = void(*)(const int8_t*, size_t, size_t, const sme::PackingParams&, void*);

template <typename GemmFn, typename PackingFn>
void run_qd8_qb4w_test(size_t M, size_t N, size_t K, PackingFn packing_fn, GemmFn gemm_fn,
                        float tol_abs, float tol_rel,
                        PackS8Fn lhs_pack_fn = nullptr,
                        PackS4Fn rhs_pack_fn = nullptr) {
  auto pack = packing_fn();

  size_t num_groups = (K + kGroupSize - 1) / kGroupSize;
  size_t K_padded = num_groups * kGroupSize;

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s4(B.data(), B.size(), 123);

  std::vector<int8_t> A_padded(M * K_padded, 0);
  for (size_t m = 0; m < M; m++)
    std::memcpy(&A_padded[m * K_padded], &A[m * K], K);

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

  std::vector<float> packed_scales(
      sme::packed_group_scales_len(num_groups, N, svl_w));
  sme::pack_group_scales(w_scales.data(), num_groups, N, svl_w,
                         packed_scales.data());

  sme::GemmParams p{M, N, K_padded};
  sme::BlockQuantParams qp{0, 0.05f, kGroupSize, packed_scales.data(), w_ksums.data()};

  std::vector<float> scratch(4 * svl_w * svl_w);

  if (lhs_pack_fn)
    lhs_pack_fn(A_padded.data(), M, K_padded, pack.lhs, lhs_packed.get());
  else
    sme::pack_s8(A_padded.data(), M, K_padded, pack.lhs, lhs_packed.get());
  if (rhs_pack_fn)
    rhs_pack_fn(B_padded.data(), K_padded, N, pack.rhs, rhs_packed.get());
  else
    sme::pack_s4(B_padded.data(), K_padded, N, pack.rhs, rhs_packed.get());
  gemm_fn(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp, scratch.data());

  sme::GemmParams p_ref{M, N, K};
  sme::BlockQuantParams qp_ref{0, 0.05f, kGroupSize, w_scales.data(), w_ksums.data()};
  sme::gemm_qd8_qb4w_f32_reference(p_ref, A.data(), B.data(), C_ref.data(), qp_ref);

  for (size_t i = 0; i < M * N; ++i) {
    float tol = tol_abs + tol_rel * std::fabs(C_ref[i]);
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

// --- 2vlxvl (2x1 SMOPA + ZA float accum) ------------------------------------

class GemmQd8Qb4w_2vlxvlTest : public ::testing::TestWithParam<GemmShape> {};

TEST_P(GemmQd8Qb4w_2vlxvlTest, MatchesReference) {
  auto [M, N, K] = GetParam();
  run_qd8_qb4w_test(M, N, K, sme::gemm_qd8_qb4w_2vlxvl_packing_params,
      [](const sme::GemmParams& p, const void* lhs, const void* rhs,
         float* out, const sme::BlockQuantParams& qp, float*) {
        sme::gemm_qd8p_qb4wp_f32_2vlxvl(p, lhs, rhs, out, qp);
      });
}

INSTANTIATE_TEST_SUITE_P(EpilogueSingleTile, GemmQd8Qb4w_2vlxvlTest,
    ::testing::Values(GemmShape{16, 16, 32}, GemmShape{16, 16, 64}, GemmShape{16, 16, 128}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(EpilogueStrides, GemmQd8Qb4w_2vlxvlTest,
    ::testing::Values(
        GemmShape{32, 16, 32}, GemmShape{16, 32, 32}, GemmShape{32, 32, 32},
        GemmShape{48, 16, 32}, GemmShape{32, 16, 64}, GemmShape{32, 32, 64}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(MainBody, GemmQd8Qb4w_2vlxvlTest,
    ::testing::Values(
        GemmShape{64, 16, 32}, GemmShape{64, 32, 32}, GemmShape{64, 16, 128},
        GemmShape{128, 16, 32}, GemmShape{128, 32, 64}, GemmShape{80, 16, 32},
        GemmShape{96, 32, 64}, GemmShape{128, 128, 128}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(PartialTiles, GemmQd8Qb4w_2vlxvlTest,
    ::testing::Values(
        GemmShape{17, 16, 32}, GemmShape{16, 17, 32}, GemmShape{16, 16, 64},
        GemmShape{17, 17, 32}, GemmShape{33, 17, 64}, GemmShape{65, 16, 32},
        GemmShape{64, 17, 32}, GemmShape{64, 16, 64}, GemmShape{65, 17, 64},
        GemmShape{80, 33, 96}, GemmShape{129, 33, 64}),
    shape_name);

// --- 2vlx2vl f16mopa (f16 widening FMOPA) ------------------------------------

class GemmQd8Qb4w_2vlx2vlF16mopaTest : public ::testing::TestWithParam<GemmShape> {};

TEST_P(GemmQd8Qb4w_2vlx2vlF16mopaTest, MatchesReference) {
  auto [M, N, K] = GetParam();
  // f16 intermediate precision requires wider tolerance than int8 SMOPA.
  run_qd8_qb4w_test(M, N, K, sme::gemm_qd8_qb4w_2vlx2vl_packing_params,
      [](const sme::GemmParams& p, const void* lhs, const void* rhs,
         float* out, const sme::BlockQuantParams& qp, float*) {
        sme::gemm_qd8p_qb4wp_f32_2vlx2vl_f16mopa(p, lhs, rhs, out, qp);
      },
      0.1f, 0.005f,
      sme::pack_s8_deinterleaved, sme::pack_s4_deinterleaved);
}

INSTANTIATE_TEST_SUITE_P(EpilogueSingleTile, GemmQd8Qb4w_2vlx2vlF16mopaTest,
    ::testing::Values(GemmShape{16, 16, 32}, GemmShape{16, 16, 64}, GemmShape{16, 16, 128}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(EpilogueStrides, GemmQd8Qb4w_2vlx2vlF16mopaTest,
    ::testing::Values(
        GemmShape{32, 16, 32}, GemmShape{16, 32, 32}, GemmShape{32, 32, 32},
        GemmShape{48, 16, 32}, GemmShape{32, 16, 64}, GemmShape{32, 32, 64}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(MainBody, GemmQd8Qb4w_2vlx2vlF16mopaTest,
    ::testing::Values(
        GemmShape{64, 16, 32}, GemmShape{64, 32, 32}, GemmShape{64, 16, 128},
        GemmShape{128, 16, 32}, GemmShape{128, 32, 64}, GemmShape{80, 16, 32},
        GemmShape{96, 32, 64}, GemmShape{128, 128, 128}),
    shape_name);

INSTANTIATE_TEST_SUITE_P(PartialTiles, GemmQd8Qb4w_2vlx2vlF16mopaTest,
    ::testing::Values(
        GemmShape{17, 16, 32}, GemmShape{16, 17, 32}, GemmShape{16, 16, 64},
        GemmShape{17, 17, 32}, GemmShape{33, 17, 64}, GemmShape{65, 16, 32},
        GemmShape{64, 17, 32}, GemmShape{64, 16, 64}, GemmShape{65, 17, 64},
        GemmShape{80, 33, 96}, GemmShape{129, 33, 64}),
    shape_name);

// --- 2vlxvl 2-level block scales (int8 inner + f32 outer) --------------------

struct GemmShape2L {
  size_t M, N, K;
  size_t inner_group_size;
  size_t outer_group_size;
};

auto shape2l_name = [](const auto& info) {
  auto s = info.param;
  return std::to_string(s.M) + "x" + std::to_string(s.K) + "x" +
         std::to_string(s.N) + "_ig" + std::to_string(s.inner_group_size) +
         "_og" + std::to_string(s.outer_group_size);
};

void fill_random_s8_range(int8_t* buf, size_t n, unsigned seed, int lo, int hi) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(lo, hi);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<int8_t>(dist(rng));
}

template <typename GemmFn, typename PackingFn>
void run_qd8_qb4w2l_test(size_t M, size_t N, size_t K,
                           size_t inner_group_size, size_t outer_group_size,
                           PackingFn packing_fn, GemmFn gemm_fn) {
  auto pack = packing_fn();

  size_t num_inner = (K + inner_group_size - 1) / inner_group_size;
  size_t num_outer = (K + outer_group_size - 1) / outer_group_size;
  size_t K_padded = num_outer * outer_group_size;

  std::vector<int8_t> A(M * K);
  std::vector<int8_t> B(K * N);
  fill_random_s8(A.data(), A.size(), 42);
  fill_random_s4(B.data(), B.size(), 123);

  // Pad to K_padded.
  std::vector<int8_t> A_padded(M * K_padded, 0);
  for (size_t m2 = 0; m2 < M; m2++)
    std::memcpy(&A_padded[m2 * K_padded], &A[m2 * K], K);

  std::vector<int8_t> B_padded(K_padded * N, 0);
  std::memcpy(B_padded.data(), B.data(), K * N);

  size_t num_inner_padded = K_padded / inner_group_size;
  size_t num_outer_padded = K_padded / outer_group_size;

  // Generate random inner scales (int8, positive, [1, 15]).
  std::vector<int8_t> inner_scales(num_inner_padded * N);
  fill_random_s8_range(inner_scales.data(), inner_scales.size(), 200, 1, 15);

  // Generate random outer scales (f32, [0.001, 0.01]).
  std::vector<float> outer_scales(num_outer_padded * N);
  fill_random_f32(outer_scales.data(), outer_scales.size(), 300, 0.001f, 0.01f);

  // Precompute ksums.
  std::vector<float> w_ksums(N);
  sme::compute_group_ksums_2level(B_padded.data(), K_padded, N,
      inner_group_size, outer_group_size,
      inner_scales.data(), outer_scales.data(), w_ksums.data());

  size_t svl_w = pack.rhs.tile_cols;

  // Tile-pack inner scales.
  std::vector<int8_t> packed_inner(
      sme::packed_group_scales_s8_len(num_inner_padded, N, svl_w));
  sme::pack_group_scales_s8(inner_scales.data(), num_inner_padded, N, svl_w,
                             packed_inner.data());

  // Tile-pack outer scales.
  std::vector<float> packed_outer(
      sme::packed_group_scales_len(num_outer_padded, N, svl_w));
  sme::pack_group_scales(outer_scales.data(), num_outer_padded, N, svl_w,
                          packed_outer.data());

  auto lhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s8(M, K_padded, pack.lhs));
  auto rhs_packed = std::make_unique<char[]>(
      sme::packed_size_bytes_s4(K_padded, N, pack.rhs));
  std::vector<float> C(M * N, 0.0f);
  std::vector<float> C_ref(M * N, 0.0f);

  sme::GemmParams p{M, N, K_padded};
  sme::BlockQuantParams2L qp{/*.a_zero_point=*/0, /*.a_scale=*/0.05f,
                              /*.inner_group_size=*/inner_group_size,
                              /*.outer_group_size=*/outer_group_size,
                              /*.inner_scales=*/packed_inner.data(),
                              /*.outer_scales=*/packed_outer.data(),
                              /*.w_ksums=*/w_ksums.data()};

  sme::pack_s8(A_padded.data(), M, K_padded, pack.lhs, lhs_packed.get());
  sme::pack_s4(B_padded.data(), K_padded, N, pack.rhs, rhs_packed.get());
  gemm_fn(p, lhs_packed.get(), rhs_packed.get(), C.data(), qp);

  // Reference uses unpacked row-major data.
  sme::GemmParams p_ref{M, N, K_padded};
  sme::BlockQuantParams2L qp_ref{0, 0.05f, inner_group_size, outer_group_size,
                                  inner_scales.data(), outer_scales.data(),
                                  w_ksums.data()};
  sme::gemm_qd8_qb4w2l_f32_reference(p_ref, A_padded.data(), B_padded.data(),
                                       C_ref.data(), qp_ref);

  for (size_t i = 0; i < M * N; ++i) {
    float tol = 2e-4f + 2e-5f * std::fabs(C_ref[i]);
    ASSERT_NEAR(C[i], C_ref[i], tol)
        << "mismatch at flat index " << i;
  }
}

class GemmQd8Qb4w2l_2vlx2vlTest : public ::testing::TestWithParam<GemmShape2L> {};

TEST_P(GemmQd8Qb4w2l_2vlx2vlTest, MatchesReference) {
  auto [M, N, K, igs, ogs] = GetParam();
  run_qd8_qb4w2l_test(M, N, K, igs, ogs,
      sme::gemm_qd8_qb4w2l_2vlx2vl_packing_params,
      [](const sme::GemmParams& p, const void* lhs, const void* rhs,
         float* out, const sme::BlockQuantParams2L& qp) {
        sme::gemm_qd8p_qb4w2lp_f32_2vlx2vl(p, lhs, rhs, out, qp);
      });
}

// inner=32, outer=128
INSTANTIATE_TEST_SUITE_P(Inner32Outer128_SmallTile, GemmQd8Qb4w2l_2vlx2vlTest,
    ::testing::Values(
        GemmShape2L{16, 16, 128, 32, 128},
        GemmShape2L{16, 16, 256, 32, 128},
        GemmShape2L{32, 16, 128, 32, 128},
        GemmShape2L{32, 32, 128, 32, 128}),
    shape2l_name);

INSTANTIATE_TEST_SUITE_P(Inner32Outer128_MainBody, GemmQd8Qb4w2l_2vlx2vlTest,
    ::testing::Values(
        GemmShape2L{64, 16, 128, 32, 128},
        GemmShape2L{128, 32, 256, 32, 128},
        GemmShape2L{128, 128, 256, 32, 128},
        GemmShape2L{128, 128, 512, 32, 128}),
    shape2l_name);

INSTANTIATE_TEST_SUITE_P(Inner32Outer128_PartialTiles, GemmQd8Qb4w2l_2vlx2vlTest,
    ::testing::Values(
        GemmShape2L{17, 16, 128, 32, 128},
        GemmShape2L{16, 17, 128, 32, 128},
        GemmShape2L{17, 17, 128, 32, 128},
        GemmShape2L{65, 17, 256, 32, 128},
        GemmShape2L{129, 33, 256, 32, 128}),
    shape2l_name);

// inner=32, outer=256
INSTANTIATE_TEST_SUITE_P(Inner32Outer256, GemmQd8Qb4w2l_2vlx2vlTest,
    ::testing::Values(
        GemmShape2L{16, 16, 256, 32, 256},
        GemmShape2L{64, 32, 256, 32, 256},
        GemmShape2L{128, 128, 512, 32, 256},
        GemmShape2L{65, 17, 512, 32, 256}),
    shape2l_name);

// inner=32, outer=512
INSTANTIATE_TEST_SUITE_P(Inner32Outer512, GemmQd8Qb4w2l_2vlx2vlTest,
    ::testing::Values(
        GemmShape2L{16, 16, 512, 32, 512},
        GemmShape2L{128, 128, 1024, 32, 512},
        GemmShape2L{65, 17, 1024, 32, 512}),
    shape2l_name);

// inner=32, outer=1024
INSTANTIATE_TEST_SUITE_P(Inner32Outer1024, GemmQd8Qb4w2l_2vlx2vlTest,
    ::testing::Values(
        GemmShape2L{16, 16, 1024, 32, 1024},
        GemmShape2L{128, 128, 2048, 32, 1024},
        GemmShape2L{65, 17, 2048, 32, 1024}),
    shape2l_name);

// inner=128, outer=128 (degenerate: 1 inner per outer)
INSTANTIATE_TEST_SUITE_P(Inner128Outer128, GemmQd8Qb4w2l_2vlx2vlTest,
    ::testing::Values(
        GemmShape2L{16, 16, 128, 128, 128},
        GemmShape2L{128, 128, 256, 128, 128},
        GemmShape2L{65, 17, 256, 128, 128}),
    shape2l_name);

// inner=128, outer=256
INSTANTIATE_TEST_SUITE_P(Inner128Outer256, GemmQd8Qb4w2l_2vlx2vlTest,
    ::testing::Values(
        GemmShape2L{16, 16, 256, 128, 256},
        GemmShape2L{128, 128, 512, 128, 256},
        GemmShape2L{65, 17, 512, 128, 256}),
    shape2l_name);

// inner=128, outer=512
INSTANTIATE_TEST_SUITE_P(Inner128Outer512, GemmQd8Qb4w2l_2vlx2vlTest,
    ::testing::Values(
        GemmShape2L{16, 16, 512, 128, 512},
        GemmShape2L{128, 128, 1024, 128, 512},
        GemmShape2L{65, 17, 1024, 128, 512}),
    shape2l_name);

// inner=128, outer=1024
INSTANTIATE_TEST_SUITE_P(Inner128Outer1024, GemmQd8Qb4w2l_2vlx2vlTest,
    ::testing::Values(
        GemmShape2L{16, 16, 1024, 128, 1024},
        GemmShape2L{128, 128, 2048, 128, 1024},
        GemmShape2L{65, 17, 2048, 128, 1024}),
    shape2l_name);

// inner=128, outer=4096
INSTANTIATE_TEST_SUITE_P(Inner128Outer4096, GemmQd8Qb4w2l_2vlx2vlTest,
    ::testing::Values(
        GemmShape2L{128, 128, 4096, 128, 4096},
        GemmShape2L{65, 17, 4096, 128, 4096}),
    shape2l_name);

}  // namespace
