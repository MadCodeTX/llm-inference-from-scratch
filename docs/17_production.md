# 17 — Production: the one-GPU home server, told truthfully

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), Ch 7, including sec. 7.2 on concurrency and batch sizing. **Production question:** what does it actually take to keep a model available, affordable, and observable for real callers — and which of those jobs does this repo's toy serving layer not do? See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Production serving is everything around the forward pass — starting replicas, sizing concurrency against a latency promise, queueing fairly, watching the right percentiles, and knowing the cost of a token — and on a single home GPU most of that machinery collapses into a short, honest checklist.

## Why inference needs this

Chapter 11 built a scheduler; chapter 16 placed engines in a stack. Neither answers the operator's questions: how long until a cold replica answers, how many concurrent callers before the p99 promise breaks, what a million tokens costs, and which graph pages you at 3 a.m. This chapter maps the book's production list onto the smallest real deployment this repo can speak about truthfully — one GPU, one model, a handful of clients — and names everything beyond that as absent rather than pretending the toy layer scales.

## Cold starts: anatomy of the wait

A fresh replica pays, in order:

1. **Process and runtime start** — interpreter, framework import, device initialisation.
2. **Weight load** — reading the checkpoint (chapter 09) and placing tensors; sharded safetensors loads per-shard, and lazy/mmap loading defers page faults into the first forwards.
3. **Cache and workspace allocation** — KV blocks (chapter 11) and kernel workspaces sized up front.
4. **Warmup** — first forwards that trigger compilation, kernel selection, and allocator steady state; these are slower than steady state and should never be served to a caller as if representative.

Measured on this machine for this repo's tiny model (fresh process, CPU wheel, median of three runs):

```text
torch_import=0.516s  pkg_import=0.001s  construct=0.003s  first_fwd=0.002s  total=0.52s
```

Framework import is essentially the entire cold start at toy scale; construction and first forward are milliseconds. That ratio inverts as models grow — weight load dominates once checkpoints reach gigabytes — which is why the anatomy matters more than this number: measure each stage separately (import, load, warmup) on the real deployment instead of quoting one total measured on a toy. The snippet is reproducible: time `import torch`, model construction, and the first forward in a fresh `.venv/bin/python -c` process, exactly as chapter 13's measurement module brackets generation.

## Concurrency, batch size, and the latency promise

Batching raises system TPS by amortising weight reads (chapters 11, 13) and simultaneously threatens per-request latency: every admitted request shares the step time, and long prompts admitted without chunking stall everyone's decode. The operator's control loop is therefore percentile-driven:

- Pick the promise first, in the vocabulary of chapter 00: for example a p95 TTFT and a p99 TPOT target for online traffic.
- Raise concurrency ([serving.py](../src/inference_from_scratch/serving.py) `max_batch_size`) while measuring both percentiles under representative prompt/output mixes ([metrics.py](../src/inference_from_scratch/metrics.py) `summarize`, [scripts/benchmark.py](../scripts/benchmark.py)).
- Stop where the tail, not the mean, touches the promise. The batch size at which system TPS saturates is usually past the batch size the SLA allows; the lower number wins for online work. Offline jobs invert the rule: saturate for aggregate TPS and let per-request latency float (chapter 00's online/offline split).

## Routing, queueing, and scale to zero — at home scale

**Routing/queueing.** With one replica there is nothing to route between; the remaining decisions are admission and order. A bounded waiting queue with FIFO admission (the toy batcher's `waiting` list) plus rejection or shedding past a queue-depth limit beats unbounded queueing, because a request that waits longer than its SLA is failed work consuming capacity. Cache-aware routing (chapter 14) only becomes real with a second replica — identical system prompts should then land on the replica holding their prefix blocks.

**Scale to zero.** Letting the replica exit when idle trades the full cold-start anatomy above for zero idle cost. On a home server the "cost" is electricity and card wear; the trade is latency honesty — the next caller pays seconds-to-minutes on a real checkpoint — against an always-warm process. The middle setting most home deployments actually want is scale-to-idle-process: weights unloaded, process and framework kept, so stage 2 alone is repaid. Whichever is chosen, the first request after idle must be labelled (or pre-warmed on a schedule) rather than silently slow.

**What this repo does not do, stated plainly:** no autoscaler, no container packaging, no multi-replica routing, no preemption, no health-checked rollout. [serving.py](../src/inference_from_scratch/serving.py) schedules requests inside one process; everything in this paragraph is operational prose around it, and chapter 16's table marks the corresponding engine capabilities accordingly.

## Cost per million tokens (symbolic)

No prices are invented here; the formula takes your own numbers:

```text
system_TPS        = measured aggregate output tokens / second (benchmark, ch 16)
tokens_per_hour   = system_TPS * 3600
cost_per_1M       = hourly_cost / tokens_per_hour * 1_000_000
                  = hourly_cost * 1_000_000 / (system_TPS * 3600)
```

`hourly_cost` is whatever the deployment actually costs per hour (amortised hardware plus electricity at home; instance price when rented). Two warnings follow from chapters 13–15: use the TPS measured at the concurrency your SLA permits, not the saturation peak, and remember decode TPS falls as context grows (KV traffic), so a cost figure is only valid for the prompt/output mix it was measured on. Quantization enters honestly through measured TPS improvement at fixed quality sign-off (chapter 14), never as an assumed multiplier.

## Observability minimum

The smallest dashboard that catches real failures, all producible from this repo's vocabulary:

- **TTFT histogram (p50/p95/p99)** — prefill health; regressions mean prompt growth, prefix-cache misses, or admission stalls.
- **TPOT / decode TPS histogram** — decode health; regressions mean over-admission, longer contexts, or bandwidth contention.
- **KV utilisation** — allocated versus free blocks (`BlockAllocator.free_blocks` in miniature); sustained near-full is the saturation early warning before latency moves.
- **Queue depth and rejection count** — demand exceeding the SLA-feasible batch, made visible instead of absorbed as silent latency.
- **Cold-start stage timings** — import, load, warmup, logged per start, so a "slow morning" is attributed in one look.

Means alone are explicitly not enough (chapters 00, 13): every latency above is a percentile or it is decoration.

## The client side

**Streaming.** Tokens should reach the caller as decode produces them; buffering a full response converts a healthy TPOT into a perceived freeze and hides stalls until the end. This repo's loops yield per-token results internally (`generate` accumulates, `measure_generate` brackets phases); a real endpoint flushes each sampled token, and TTFT is precisely the wait for the first flush.

**Asynchronous and offline work.** Batch jobs (summarisation, eval sweeps) should be submitted as jobs with completion polling or callbacks, run at saturation batch sizes, and kept off the online queue — the offline pole of chapter 00. Mixing the two classes in one FIFO is how a bulk job destroys an interactive p99.

## Walk the code

- Scheduler and admission: [serving.py](../src/inference_from_scratch/serving.py) `ContinuousBatcher` (`add`/`step`/`run`), `BlockAllocator` for KV utilisation.
- Percentiles and phase timing: [metrics.py](../src/inference_from_scratch/metrics.py) `summarize`, `measure_generate`; [scripts/benchmark.py](../scripts/benchmark.py) as the runnable harness.
- Cold-start staging: model construction and loading in [model.py](../src/inference_from_scratch/model.py) and [weights.py](../src/inference_from_scratch/weights.py) (ch 09), timed in a fresh process as above.

## How the big systems do it

Fleet production adds what one GPU cannot rehearse: autoscaling on queue depth and KV utilisation, containerised replicas with staged rollouts and output-quality gates (chapter 14's metrics as release criteria), cache-aware load balancing, and prefill/decode pools scaled independently (disaggregation, chapter 14). The observability minimum and the cost formula survive unchanged — they are per-replica truths that fleets aggregate.

## Common pitfalls

- **Sizing batch by saturation TPS.** The SLA-feasible batch is smaller; the tail decides (see above).
- **Unbounded queues.** Latency debt disguised as reliability. Bound, shed, and count rejections.
- **Quoting cold toys as cold truth.** Stage timings from a 78k-parameter model describe the anatomy, not a real checkpoint's minutes.
- **Averaging away the incident.** A healthy mean with a broken p99 is a broken service that graphs well.

## Break it / exercises

1. Reproduce the cold-start staging snippet in a fresh process three times. Which stage varies most run to run, and what would dominate instead with a multi-gigabyte checkpoint?
2. Using a symbolic `hourly_cost = C`, compute `cost_per_1M` at two measured TPS values from `scripts/benchmark.py` (saturation versus half batch). State the SLA caveat in one sentence.
3. Add (on paper) a second replica. Which single routing rule captures most of chapter 14's prefix benefit, and which observability item tells you it is working?

## Check yourself

<details>
<summary>1. Why is the SLA-feasible batch usually smaller than the throughput-maximising batch?</summary>

System TPS keeps rising as weight reads amortise, but shared step time and admission stalls degrade per-request TPOT and TTFT tails. The promise is written in percentiles, so the tail — not peak throughput — sets the limit.
</details>

<details>
<summary>2. A service scales to zero overnight. What exactly does the first morning request pay?</summary>

The full cold-start anatomy: process/framework start, weight load, cache allocation, and warmup forwards — in that order, and measurably per stage.
</details>

<details>
<summary>3. Which single metric warns of KV saturation before latency degrades?</summary>

KV utilisation (allocated versus free blocks). Near-full allocation precedes admission stalls; TTFT/TPOT percentiles only move afterwards.
</details>

## Summary in 30 seconds

Promise percentiles first, size concurrency to the tail, bound the queue, price tokens from measured TPS, and graph TTFT/TPOT/KV from day one. If you remember one formula, make it `cost_per_1M = hourly_cost · 10⁶ / (TPS · 3600)` at SLA-feasible TPS; if you remember one code location, make it `summarize` in [metrics.py](../src/inference_from_scratch/metrics.py).

## Where to go next

Continue to ch 18 (modalities), the read-only map of what changes beyond text. Run `.venv/bin/python -m pytest tests/test_serving.py tests/test_metrics.py -q` before moving on.
