# MOPA Peak Throughput — Qualcomm (Snapdragon 8 Elite)

Device: Qualcomm Snapdragon 8 Elite (adb serial `7cd0501`)
SVL: 64 bytes, VL (f32): 16 elements

## Results

| Instruction | Rank | Ops/MOPA | Throughput (G MOPA/s) | Peak (GOP/s) |
|---|---|---|---|---|
| f32 FMOPA | 1 | 512 | 1.93 | 990 |
| bf16 BFMOPA | 2 | 1024 | 1.93 | 1980 |
| f16 FMOPA | 2 | 1024 | 1.93 | 1980 |
| i8 SMOPA | 4 | 2048 | 3.86 | 7913 |
| i8 s4-unpack+SMOPA | 4 | 2048 | 3.86 | 7911 |

## Observations

- f32 FMOPA sustains ~1.93 G MOPA/s (~990 GFLOP/s).
- bf16 BFMOPA and f16 FMOPA are identical at ~1.93 G MOPA/s, achieving 2x GOP/s due to rank-2 outer product (~1980 GFLOP/s).
- i8 SMOPA sustains ~3.86 G MOPA/s (2x the f32 rate), with 4x ops per instruction → ~8 TOP/s peak.
- s4 nibble unpack (shift+mask) adds zero measurable overhead; fully hidden behind SMOPA latency.
