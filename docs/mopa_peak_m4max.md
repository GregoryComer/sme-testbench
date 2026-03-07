# MOPA Peak Throughput — Apple M4 Max

Device: Apple M4 Max
SVL: 64 bytes, VL (f32): 16 elements
Cluster: P0 (6 cores, max 4512 MHz)
Measured frequency: ~4.47 GHz (per-core HW active freq during benchmark)

## Results

| Instruction | Rank | Ops/MOPA | Peak (GOP/s) | MOPA/s | MOPA/cycle | Cycles/MOPA |
|---|---|---|---|---|---|---|
| f32 FMOPA | 1 | 512 | 1989 | 3.89 G | 0.870 | 1.15 |
| bf16 BFMOPA | 2 | 1024 | 1985 | 1.94 G | 0.434 | 2.30 |
| f16 FMOPA | 2 | 1024 | 1979 | 1.93 G | 0.432 | 2.31 |
| i8 SMOPA | 4 | 2048 | 3966 | 1.94 G | 0.434 | 2.30 |
| i8 s4-unpack+SMOPA | 4 | 2048 | 3955 | 1.93 G | 0.432 | 2.31 |

## Observations

- f32 FMOPA sustains ~0.87 MOPA/cycle (~1.15 cycles/MOPA), nearly saturating at 1 MOPA/cycle. This yields ~2.0 TFLOP/s f32.
- bf16 BFMOPA, f16 FMOPA, and i8 SMOPA all share the same throughput: ~0.43 MOPA/cycle (~2.3 cycles/MOPA). The widening variants appear to use the same pipeline with no additional throughput advantage.
- s4 nibble unpack adds zero measurable overhead; fully hidden behind MOPA latency.

## Comparison with Qualcomm Snapdragon 8 Elite (prime core)

| Instruction | M4 Max (4.47 GHz) | Qualcomm prime (4.40 GHz) |
|---|---|---|
| | MOPA/cycle / GOP/s | MOPA/cycle / GOP/s |
| f32 FMOPA | 0.870 / 1989 | 0.439 / 990 |
| bf16 BFMOPA | 0.434 / 1985 | 0.439 / 1980 |
| f16 FMOPA | 0.432 / 1979 | 0.439 / 1980 |
| i8 SMOPA | 0.434 / 3966 | 0.879 / 7913 |

Key differences:
- **f32**: M4 is 2x faster per-MOPA (0.87 vs 0.44 MOPA/cycle), giving 2x the f32 FLOP/s.
- **bf16/f16**: Identical throughput (~1980 GOP/s) on both — M4's 2x f32 advantage disappears for rank-2 widening variants.
- **i8**: Qualcomm is 2x faster per-MOPA (0.88 vs 0.43 MOPA/cycle), giving 2x the i8 TOP/s (~8 vs ~4). Qualcomm issues SMOPA at 2x the rate of its own FMOPA; M4 does not.
- M4 has a fast path for f32 rank-1 FMOPA (~1 MOPA/cycle). All other variants share a ~2.3 cycle pipeline.
- Qualcomm has a fast path for i8 rank-4 SMOPA (~1.14 cycles). f32/bf16/f16 share a ~2.28 cycle pipeline.
