#include "gemm.h"

// Forward-declare SME kernels without streaming attributes so this TU
// compiles without -march=+sme.  The actual symbols are defined in the
// *_sme.cpp files with the proper __arm_streaming __arm_inout("za").
namespace sme {

// 4vlxvl (4x1 tile layout)
void gemm_f32p_f32p_f32_4vlxvl_kernel(const GemmParams& p, const void* lhs_packed,
                                      const void* rhs_packed, float* out);
void gemm_f16p_f16p_f16_4vlxvl_kernel(const GemmParams& p, const void* lhs_packed,
                                      const void* rhs_packed, _Float16* out);
void gemm_bf16p_bf16p_bf16_4vlxvl_kernel(const GemmParams& p, const void* lhs_packed,
                                         const void* rhs_packed, __bf16* out);
void gemm_qd8p_qc8wp_f32_4vlxvl_kernel(const GemmParams& p, const void* lhs_packed,
                                        const void* rhs_packed, float* out,
                                        const QuantParams& qp);

// 2vlx2vl (2x2 tile layout)
void gemm_f32p_f32p_f32_2vlx2vl_kernel(const GemmParams& p, const void* lhs_packed,
                                      const void* rhs_packed, float* out);
void gemm_f16p_f16p_f16_2vlx2vl_kernel(const GemmParams& p, const void* lhs_packed,
                                      const void* rhs_packed, _Float16* out);
void gemm_bf16p_bf16p_bf16_2vlx2vl_kernel(const GemmParams& p, const void* lhs_packed,
                                         const void* rhs_packed, __bf16* out);
void gemm_qd8p_qc8wp_f32_2vlx2vl_kernel(const GemmParams& p, const void* lhs_packed,
                                        const void* rhs_packed, float* out,
                                        const QuantParams& qp);

// qc4w 4vlxvl
void gemm_qd8p_qc4wp_f32_4vlxvl_kernel(const GemmParams& p, const void* lhs_packed,
                                        const void* rhs_packed, float* out,
                                        const QuantParams& qp);

// qb4w 4vlxvl
void gemm_qd8p_qb4wp_f32_4vlxvl_kernel(const GemmParams& p, const void* lhs_packed,
                                        const void* rhs_packed, float* out,
                                        const BlockQuantParams& qp,
                                        float* scratch);

// qb4w 2vlx2vl
void gemm_qd8p_qb4wp_f32_2vlx2vl_kernel(const GemmParams& p, const void* lhs_packed,
                                         const void* rhs_packed, float* out,
                                         const BlockQuantParams& qp);

// qb4w 2vlxvl (ZA float accumulation)
void gemm_qd8p_qb4wp_f32_2vlxvl_kernel(const GemmParams& p, const void* lhs_packed,
                                        const void* rhs_packed, float* out,
                                        const BlockQuantParams& qp);

// qb4w 2vlx2vl f16mopa (locally streaming, called from non-streaming orchestrator)
void gemm_qd8p_qb4wp_f32_2vlx2vl_f16mopa_kernel(const GemmParams& p, const void* lhs_packed,
                                                  const void* rhs_packed, float* out,
                                                  const BlockQuantParams& qp);
}

namespace sme {

// ---------- f32 4vlxvl --------------------------------------------------------

void gemm_f32p_f32p_f32_4vlxvl(const GemmParams& p, const void* lhs_packed,
                               const void* rhs_packed, float* out) {
  asm volatile("smstart" ::: "memory");
  gemm_f32p_f32p_f32_4vlxvl_kernel(p, lhs_packed, rhs_packed, out);
  asm volatile("smstop" ::: "memory");
}

// ---------- f32 2vlx2vl --------------------------------------------------------

void gemm_f32p_f32p_f32_2vlx2vl(const GemmParams& p, const void* lhs_packed,
                               const void* rhs_packed, float* out) {
  asm volatile("smstart" ::: "memory");
  gemm_f32p_f32p_f32_2vlx2vl_kernel(p, lhs_packed, rhs_packed, out);
  asm volatile("smstop" ::: "memory");
}

// ---------- f16 4vlxvl --------------------------------------------------------

void gemm_f16p_f16p_f16_4vlxvl(const GemmParams& p, const void* lhs_packed,
                               const void* rhs_packed, _Float16* out) {
  asm volatile("smstart" ::: "memory");
  gemm_f16p_f16p_f16_4vlxvl_kernel(p, lhs_packed, rhs_packed, out);
  asm volatile("smstop" ::: "memory");
}

// ---------- f16 2vlx2vl --------------------------------------------------------

void gemm_f16p_f16p_f16_2vlx2vl(const GemmParams& p, const void* lhs_packed,
                               const void* rhs_packed, _Float16* out) {
  asm volatile("smstart" ::: "memory");
  gemm_f16p_f16p_f16_2vlx2vl_kernel(p, lhs_packed, rhs_packed, out);
  asm volatile("smstop" ::: "memory");
}

// ---------- bf16 4vlxvl -------------------------------------------------------

void gemm_bf16p_bf16p_bf16_4vlxvl(const GemmParams& p, const void* lhs_packed,
                                  const void* rhs_packed, __bf16* out) {
  asm volatile("smstart" ::: "memory");
  gemm_bf16p_bf16p_bf16_4vlxvl_kernel(p, lhs_packed, rhs_packed, out);
  asm volatile("smstop" ::: "memory");
}

// ---------- bf16 2vlx2vl -------------------------------------------------------

void gemm_bf16p_bf16p_bf16_2vlx2vl(const GemmParams& p, const void* lhs_packed,
                                  const void* rhs_packed, __bf16* out) {
  asm volatile("smstart" ::: "memory");
  gemm_bf16p_bf16p_bf16_2vlx2vl_kernel(p, lhs_packed, rhs_packed, out);
  asm volatile("smstop" ::: "memory");
}

// ---------- qd8 4vlxvl -------------------------------------------------------

void gemm_qd8p_qc8wp_f32_4vlxvl(const GemmParams& p, const void* lhs_packed,
                                 const void* rhs_packed, float* out,
                                 const QuantParams& qp) {
  asm volatile("smstart" ::: "memory");
  gemm_qd8p_qc8wp_f32_4vlxvl_kernel(p, lhs_packed, rhs_packed, out, qp);
  asm volatile("smstop" ::: "memory");
}

// ---------- qd8 2vlx2vl -------------------------------------------------------

void gemm_qd8p_qc8wp_f32_2vlx2vl(const GemmParams& p, const void* lhs_packed,
                                 const void* rhs_packed, float* out,
                                 const QuantParams& qp) {
  asm volatile("smstart" ::: "memory");
  gemm_qd8p_qc8wp_f32_2vlx2vl_kernel(p, lhs_packed, rhs_packed, out, qp);
  asm volatile("smstop" ::: "memory");
}

// ---------- qc4w 4vlxvl -------------------------------------------------------

void gemm_qd8p_qc4wp_f32_4vlxvl(const GemmParams& p, const void* lhs_packed,
                                 const void* rhs_packed, float* out,
                                 const QuantParams& qp) {
  asm volatile("smstart" ::: "memory");
  gemm_qd8p_qc4wp_f32_4vlxvl_kernel(p, lhs_packed, rhs_packed, out, qp);
  asm volatile("smstop" ::: "memory");
}

// ---------- qb4w 4vlxvl -------------------------------------------------------

void gemm_qd8p_qb4wp_f32_4vlxvl(const GemmParams& p, const void* lhs_packed,
                                 const void* rhs_packed, float* out,
                                 const BlockQuantParams& qp,
                                 float* scratch) {
  asm volatile("smstart" ::: "memory");
  gemm_qd8p_qb4wp_f32_4vlxvl_kernel(p, lhs_packed, rhs_packed, out, qp, scratch);
  asm volatile("smstop" ::: "memory");
}

// ---------- qb4w 2vlx2vl ------------------------------------------------------

void gemm_qd8p_qb4wp_f32_2vlx2vl(const GemmParams& p, const void* lhs_packed,
                                  const void* rhs_packed, float* out,
                                  const BlockQuantParams& qp) {
  asm volatile("smstart" ::: "memory");
  gemm_qd8p_qb4wp_f32_2vlx2vl_kernel(p, lhs_packed, rhs_packed, out, qp);
  asm volatile("smstop" ::: "memory");
}

// ---------- qb4w 2vlxvl (ZA float accumulation) ------------------------------

void gemm_qd8p_qb4wp_f32_2vlxvl(const GemmParams& p, const void* lhs_packed,
                                 const void* rhs_packed, float* out,
                                 const BlockQuantParams& qp) {
  asm volatile("smstart" ::: "memory");
  gemm_qd8p_qb4wp_f32_2vlxvl_kernel(p, lhs_packed, rhs_packed, out, qp);
  asm volatile("smstop" ::: "memory");
}

// ---------- qb4w 2vlx2vl f16mopa (f16 widening FMOPA) -------------------------

void gemm_qd8p_qb4wp_f32_2vlx2vl_f16mopa(const GemmParams& p, const void* lhs_packed,
                                            const void* rhs_packed, float* out,
                                            const BlockQuantParams& qp) {
  gemm_qd8p_qb4wp_f32_2vlx2vl_f16mopa_kernel(p, lhs_packed, rhs_packed, out, qp);
}

}  // namespace sme
