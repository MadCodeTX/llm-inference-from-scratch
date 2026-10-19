# Library — decode

Index of adopted/rejected decode levers. Read before proposing on the
`decode-tpot` card. Full entries: [EXPERIENCE.md](EXPERIENCE.md).

| Entry | Cue | Verdict |
|---|---|---|
| D1 reference-dequant bottleneck | bridge decode ~1.8 tok/s | superseded by fused FP8 (D2) |
| D2 fused FP8 (Triton, default) | dequant dominates profile | adopted (bridge v2) |
| D3 host-launch gap → CUDA graph | clean wall ≫ GPU busy | adopted (graph decode) |
| D4 BLOCK_N tile retune after graph | FP8 kernel top cost under graph | adopted (`--fp8-config new`) |
| D5 fused gated-delta | elementwise/delta cost under graph | adopted (`--delta fused`, best 23.67 ms) |
| D6 scale-cache graph-only gain | graphed decode, single-variable probe | not adopted — base retained pending confirmation |
