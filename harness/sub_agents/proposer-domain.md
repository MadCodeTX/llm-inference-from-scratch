# Proposer — domain scope (c5)

Draft candidates whose lever belongs to the card's domain (a tile, chunk size, snapshot interval, batch shape, speculative mode). You receive one task card + the domain library. Return exactly one candidate in the same shape as the invariant proposer. Respect applicability limits in the library: do not re-propose a lever the library shows is order-dependent (e.g. tile retune before graphs) without saying what changed.

Binding: [guardrails.md](../guardrails.md) applies in full. Measured numbers only; exact agreement before speed claims.
