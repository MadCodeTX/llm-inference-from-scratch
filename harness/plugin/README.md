# Plugin (c8) — tool-loop enforcement

No plugin ships with this tree, deliberately: every rule in
[`../guardrails.md`](../guardrails.md) is currently enforced by the probe
runners themselves (the TP2 runner records `exact_match` per probe; the
probe runner role executes only card-named commands).

A plugin becomes the right lever only for a recurring, mechanical failure
that prose keeps failing to prevent — the known candidate shapes, in order
of expected value:

1. **Exactness tripwire**: after any probe artifact is written, check
   `exact_match` / `failed` fields and block the step from composing until
   inexact candidates are marked rejected.
2. **Cross-shape claim lint**: flag any draft text that compares a
   local-runner number against a `run_bench.py` bar (or vice versa) as a
   win/loss.
3. **GPU-window check**: refuse a live `run_bench.py` invocation outside an
   approved window marker.

Until one of these failures recurs measurably, c8 stays empty: an
unenforced rule that is actually followed is cheaper than machinery.
