# Library — prefill

Read before proposing on the `prefill-ttft` card. Entries: [EXPERIENCE.md](EXPERIENCE.md).

| Entry | Cue | Verdict |
|---|---|---|
| P1 chunkwise prefill + dispatch threshold | token-by-token prefill ~23 tok/s | adopted (chunk512 best @2048) |
| P2 prefill num_stages=2 (prefill-s2) | unmeasured stages axis | rejected (-1.3% to -7.5% everywhere; code removed) |
