// Peak MOPA throughput microbenchmark.
// 64 back-to-back mopa instructions per loop iteration (4 ZA tiles × 16 reps),
// with varied source registers. No loads/stores in the hot loop.
//
// Tests three variants:
//   f32  FMOPA: za.s += z.s * z.s     (rank-1,  2*VL²    FLOPs/mopa)
//   bf16 FMOPA: za.s += z.h * z.h     (rank-2,  4*VL²    FLOPs/mopa)
//   f16  FMOPA: za.s += z.h * z.h     (rank-2,  4*VL²    FLOPs/mopa)
//   i8   SMOPA: za.s += z.b * z.b     (rank-4,  8*VL²    OPs/mopa)

#include <chrono>
#include <cstdint>
#include <cstdio>

// ---- f32 FMOPA (rank-1) ---------------------------------------------------
#define F32_MOPA4                                \
  "fmopa za0.s, p0/m, p0/m, z0.s, z4.s\n"      \
  "fmopa za1.s, p0/m, p0/m, z1.s, z5.s\n"      \
  "fmopa za2.s, p0/m, p0/m, z2.s, z6.s\n"      \
  "fmopa za3.s, p0/m, p0/m, z3.s, z7.s\n"
#define F32_MOPA16 F32_MOPA4 F32_MOPA4 F32_MOPA4 F32_MOPA4
#define F32_MOPA64 F32_MOPA16 F32_MOPA16 F32_MOPA16 F32_MOPA16

static void run_f32(uint64_t iterations) {
  asm volatile(
      "smstart\n"
      "ptrue p0.s\n"
      "zero {za}\n"
      "fmov z0.s, #1.0\n"
      "fmov z1.s, #1.0\n"
      "fmov z2.s, #1.0\n"
      "fmov z3.s, #1.0\n"
      "fmov z4.s, #1.0\n"
      "fmov z5.s, #1.0\n"
      "fmov z6.s, #1.0\n"
      "fmov z7.s, #1.0\n"
      "1:\n"
      F32_MOPA64
      "subs %[n], %[n], #1\n"
      "b.ne 1b\n"
      "smstop\n"
      : [n] "+r"(iterations)
      :
      : "cc", "memory",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15");
}

// ---- bf16→f32 widening FMOPA (rank-2) -------------------------------------
#define BF16_MOPA4                               \
  "bfmopa za0.s, p0/m, p0/m, z0.h, z4.h\n"     \
  "bfmopa za1.s, p0/m, p0/m, z1.h, z5.h\n"     \
  "bfmopa za2.s, p0/m, p0/m, z2.h, z6.h\n"     \
  "bfmopa za3.s, p0/m, p0/m, z3.h, z7.h\n"
#define BF16_MOPA16 BF16_MOPA4 BF16_MOPA4 BF16_MOPA4 BF16_MOPA4
#define BF16_MOPA64 BF16_MOPA16 BF16_MOPA16 BF16_MOPA16 BF16_MOPA16

static void run_bf16(uint64_t iterations) {
  asm volatile(
      "smstart\n"
      "ptrue p0.h\n"
      "zero {za}\n"
      "mov z0.h, #0x3f80\n"
      "mov z1.h, #0x3f80\n"
      "mov z2.h, #0x3f80\n"
      "mov z3.h, #0x3f80\n"
      "mov z4.h, #0x3f80\n"
      "mov z5.h, #0x3f80\n"
      "mov z6.h, #0x3f80\n"
      "mov z7.h, #0x3f80\n"
      "1:\n"
      BF16_MOPA64
      "subs %[n], %[n], #1\n"
      "b.ne 1b\n"
      "smstop\n"
      : [n] "+r"(iterations)
      :
      : "cc", "memory",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15");
}

// ---- f16→f32 widening FMOPA (rank-2) --------------------------------------
#define F16_MOPA4                                \
  "fmopa za0.s, p0/m, p0/m, z0.h, z4.h\n"      \
  "fmopa za1.s, p0/m, p0/m, z1.h, z5.h\n"      \
  "fmopa za2.s, p0/m, p0/m, z2.h, z6.h\n"      \
  "fmopa za3.s, p0/m, p0/m, z3.h, z7.h\n"
#define F16_MOPA16 F16_MOPA4 F16_MOPA4 F16_MOPA4 F16_MOPA4
#define F16_MOPA64 F16_MOPA16 F16_MOPA16 F16_MOPA16 F16_MOPA16

static void run_f16(uint64_t iterations) {
  asm volatile(
      "smstart\n"
      "ptrue p0.h\n"
      "zero {za}\n"
      "fmov z0.h, #1.0\n"
      "fmov z1.h, #1.0\n"
      "fmov z2.h, #1.0\n"
      "fmov z3.h, #1.0\n"
      "fmov z4.h, #1.0\n"
      "fmov z5.h, #1.0\n"
      "fmov z6.h, #1.0\n"
      "fmov z7.h, #1.0\n"
      "1:\n"
      F16_MOPA64
      "subs %[n], %[n], #1\n"
      "b.ne 1b\n"
      "smstop\n"
      : [n] "+r"(iterations)
      :
      : "cc", "memory",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15");
}

// ---- i8→i32 widening SMOPA (rank-4) ---------------------------------------
#define I8_MOPA4                                 \
  "smopa za0.s, p0/m, p0/m, z0.b, z4.b\n"      \
  "smopa za1.s, p0/m, p0/m, z1.b, z5.b\n"      \
  "smopa za2.s, p0/m, p0/m, z2.b, z6.b\n"      \
  "smopa za3.s, p0/m, p0/m, z3.b, z7.b\n"
#define I8_MOPA16 I8_MOPA4 I8_MOPA4 I8_MOPA4 I8_MOPA4
#define I8_MOPA64 I8_MOPA16 I8_MOPA16 I8_MOPA16 I8_MOPA16

static void run_i8(uint64_t iterations) {
  asm volatile(
      "smstart\n"
      "ptrue p0.b\n"
      "zero {za}\n"
      "mov z0.b, #1\n"
      "mov z1.b, #1\n"
      "mov z2.b, #1\n"
      "mov z3.b, #1\n"
      "mov z4.b, #1\n"
      "mov z5.b, #1\n"
      "mov z6.b, #1\n"
      "mov z7.b, #1\n"
      "1:\n"
      I8_MOPA64
      "subs %[n], %[n], #1\n"
      "b.ne 1b\n"
      "smstop\n"
      : [n] "+r"(iterations)
      :
      : "cc", "memory",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15");
}

// ---- i8 s4-unpack + SMOPA (3 shifts + 8 MOPAs per block) -----------------
// Measures the throughput cost of interleaving the nibble-unpack sequence
// (LSL #4, ASR #4, ASR #4) with SMOPA.  64 MOPAs per iteration (8 blocks).
#define S4_BLOCK                                       \
  "lsl z9.b, z8.b, #4\n"                              \
  "asr z10.b, z9.b, #4\n"                             \
  "asr z11.b, z8.b, #4\n"                             \
  "smopa za0.s, p0/m, p0/m, z0.b, z10.b\n"            \
  "smopa za1.s, p0/m, p0/m, z1.b, z10.b\n"            \
  "smopa za2.s, p0/m, p0/m, z2.b, z10.b\n"            \
  "smopa za3.s, p0/m, p0/m, z3.b, z10.b\n"            \
  "smopa za0.s, p0/m, p0/m, z4.b, z11.b\n"            \
  "smopa za1.s, p0/m, p0/m, z5.b, z11.b\n"            \
  "smopa za2.s, p0/m, p0/m, z6.b, z11.b\n"            \
  "smopa za3.s, p0/m, p0/m, z7.b, z11.b\n"
#define S4_BLOCK8 S4_BLOCK S4_BLOCK S4_BLOCK S4_BLOCK \
                  S4_BLOCK S4_BLOCK S4_BLOCK S4_BLOCK

static void run_i8_s4_unpack(uint64_t iterations) {
  asm volatile(
      "smstart\n"
      "ptrue p0.b\n"
      "zero {za}\n"
      "mov z0.b, #1\n"
      "mov z1.b, #1\n"
      "mov z2.b, #1\n"
      "mov z3.b, #1\n"
      "mov z4.b, #1\n"
      "mov z5.b, #1\n"
      "mov z6.b, #1\n"
      "mov z7.b, #1\n"
      "mov z8.b, #0x37\n"
      "1:\n"
      S4_BLOCK8
      "subs %[n], %[n], #1\n"
      "b.ne 1b\n"
      "smstop\n"
      : [n] "+r"(iterations)
      :
      : "cc", "memory",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15");
}

// ---------------------------------------------------------------------------

struct Variant {
  const char* name;
  void (*fn)(uint64_t);
  int rank;  // ops per accumulator element per mopa
};

int main() {
  uint64_t svl_bytes;
  asm("rdsvl %0, #1" : "=r"(svl_bytes));
  uint64_t vl = svl_bytes / sizeof(float);  // VL in f32 elements

  printf("SVL: %llu bytes, VL (f32): %llu\n",
         (unsigned long long)svl_bytes, (unsigned long long)vl);
  printf("\n");

  constexpr uint64_t mopas_per_iter = 64;

  Variant variants[] = {
      {"f32  FMOPA (rank-1)",       run_f32,           1},
      {"bf16 BFMOPA (rank-2)",      run_bf16,          2},
      {"f16  FMOPA (rank-2)",       run_f16,           2},
      {"i8   SMOPA (rank-4)",       run_i8,            4},
      {"i8   s4-unpack+SMOPA",      run_i8_s4_unpack,  4},
  };

  for (auto& v : variants) {
    // ops per mopa = 2 * rank * VL² (2 for multiply+accumulate)
    double ops_per_mopa = 2.0 * v.rank * vl * vl;

    printf("--- %s --- (%.0f ops/mopa)\n", v.name, ops_per_mopa);

    // Warm up
    v.fn(10000);

    for (uint64_t iters :
         {1000000UL, 10000000UL, 50000000UL, 200000000UL}) {
      auto start = std::chrono::high_resolution_clock::now();
      v.fn(iters);
      auto end = std::chrono::high_resolution_clock::now();

      double secs = std::chrono::duration<double>(end - start).count();
      double total_mopas = static_cast<double>(iters) * mopas_per_iter;
      double gops = total_mopas * ops_per_mopa / secs / 1e9;

      printf("  Iters: %8lu | %7.4f s | %5.2f G MOPA/s | %7.1f GOP/s\n",
             (unsigned long)iters, secs, total_mopas / secs / 1e9, gops);
    }
    printf("\n");
  }

  return 0;
}
