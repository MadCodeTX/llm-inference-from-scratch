# Textbook map — how this repo uses *Inference Engineering*

> Philip Kiely, *Inference Engineering*, Baseten Books, January 2026,
> ISBN 9798994359730.
>
> Framing follows Kiely, *Inference Engineering* (2026). Section numbers
> below refer to that book.

## How this repo uses the book

This repo borrows the book's framing, categorisation, section numbering,
and core terminology — for example the prefill/decode split, TTFT and TPS,
percentiles, and the compute-bound versus memory-bound distinction.
All prose, code, worked examples, and tests here are original to this
repo. Nothing is copied from the book beyond short technical terms and
section titles. Where a chapter leans on the book's framing, its header
says so in the form: *Framing follows Kiely, Inference Engineering
(2026), sec. X.Y.*

The book is a production survey. This repo is a dissection kit. The book
asks what a deployed system should do; each chapter here builds the
smallest hand-written version of one mechanism, tests it at tiny size,
and only then points at the production counterpart. Reading the two
together: use the book for the decision, use this repo for the mechanism
underneath it.

New readers: start at [00_overview.md](00_overview.md), keep
[GLOSSARY.md](GLOSSARY.md) open, and use [ROADMAP.md](ROADMAP.md) for
the grouped curriculum. This file is the cross-reference, not a chapter.

## Chapter map: repo 00–18 to book sections

Build status is stated plainly. **Built** means hand-written code plus
tests exist today. **Read-only** means the chapter dissects production
code paths without integrating them. **Planned** means only this map and
the roadmap describe it so far — no chapter file or code exists yet.

| Repo ch | Book section(s) | Status | Book question this chapter answers |
|---------|-----------------|--------|-----------------------------------|
| [00 Overview](00_overview.md) | Ch 0; sec. 1.2, 1.4 | Built | What happens between a request arriving and tokens streaming back, and how should latency be measured? |
| [01 Tokenizer](01_tokenizer.md) | Sec. 2.2 (lightly; see gap note) | Built | How does raw text become the integer ids the model actually consumes? |
| [02 Embeddings + RoPE](02_embeddings_rope.md) | Sec. 2.2.1–2.2.2 | Built | How do ids get meaning and position before attention runs? |
| [03 Norms](03_norms.md) | Sec. 2.2.2 | Built | How does a deep stack keep activation scale under control? |
| [04 Attention](04_attention.md) | Sec. 2.2.3, 2.5 | Built | How do tokens exchange information without seeing the future? |
| [05 KV cache](05_kv_cache.md) | Sec. 2.5, 5.3 | Built | Why is past work stored instead of recomputed, and what does that storage cost? |
| [06 MLP](06_mlp.md) | Sec. 2.1.2, 2.2.2; MoE pointer sec. 2.2.4 | Built | Where do most parameters live, and what transforms a single token? |
| [07 Model](07_model.md) | Sec. 2.2.1 | Built | How do the parts assemble into one decoder driven by `config.json`? |
| [08 Sampling / generation](08_sampling_generation.md) | Sec. 2.2 (autoregressive mechanics) | Built | How does a probability distribution become a stream of tokens? |
| [09 Weights](09_weights.md) | Sec. 4.2.2, 4.2.4 | Built | How do bytes on disk become a runnable model with the right shapes? |
| [10 Quantization](10_quantization.md) | Sec. 5.1 | Built | How few bits per weight can we store before quality suffers, and what do we gain? |
| [11 Serving](11_serving.md) | Sec. 4.3, 7.2 | Built | How do many concurrent requests share one model without wasting the hardware? |
| [12 Optimizations](12_optimizations.md) | Sec. 4.1; Ch 5 overview | Read-only | Which fast path replaces which slow mechanism, and how would we know it is still correct? |
| [13 Bottlenecks](13_bottlenecks.md) | Sec. 2.4, 1.4 | Built | For a given model and device, is arithmetic or data movement the limit — and by how much? |
| [14 Techniques: prefix cache + speculation](14_techniques.md) | Sec. 5.3.1, 5.2, 5.1.3, 5.4–5.5 | Built | How can shared prefixes and drafted tokens cut work without changing outputs, and how is a deliberate quality trade measured? |
| [15 Hardware](15_hardware.md) | Ch 3 (incl. sec. 3.5) | Written | What do GPU compute, memory tiers, and local devices allow this code to do? |
| [16 Software / engines](16_software_engines.md) | Ch 4 (incl. sec. 4.2.4, 4.5) | Written | Where does this hand-built engine sit beside vLLM, SGLang, TensorRT-LLM, and Dynamo? |
| [17 Production](17_production.md) | Ch 7 (incl. sec. 7.2) | Written | What do cold starts, autoscaling, cost, observability, and client streaming add around the model? |
| [18 Modalities](18_modalities.md) | Ch 6 | Written, read-only | What changes when inputs or outputs stop being plain text? |

Chapters 13–18 are written. Chapters 15–17 are prose chapters over
existing code (no new code); chapter 18 is a read-only map with no
code by design.

## Where this repo deliberately goes deeper than the book

Three topics get a slower, byte-level treatment here than a production
survey can afford:

- **Tokenizer and BPE mechanics (ch 01).** The book touches tokenisation
  lightly on its way to serving decisions. Here the merge loop is built
  by hand, roundtrip losslessness is tested, and the length-for-vocab
  tradeoff is priced in prefill FLOPs and KV bytes. That depth is a gap
  this repo fills, not a disagreement.
- **RoPE mathematics (ch 02).** Rotation angles, the rotate-half layout,
  norm preservation, and the relative-distance property are derived and
  tested property by property, because a wrong pairing convention fails
  silently at generation time.
- **Safetensors bytes (ch 09).** The 8-byte header length, JSON header,
  and offset arithmetic are parsed by hand and validated entry by entry,
  so loading a real checkpoint is a renaming exercise a reader can audit.

## Book areas out of scope for now

Deliberately not covered, until the planned chapters say otherwise:

- **Image and video generation** (book sec. 2.3, 6.5, 6.6). Chapter 18
  maps the area read-only; no diffusion or video code is planned.
- **Multi-cloud capacity and procurement** (book sec. 7.3). Chapter 17
  covers only what this repo's toy serving layer does and does not do.
- **Dynamo operations at fleet scale** (book sec. 4.4, 5.5.3).
  Chapter 16 compares engines; it does not operate a disaggregated
  fleet. Disaggregation itself is defined in the glossary and mapped
  in chapter 14/17 planning notes.
- **Managed container products** such as NIMs (book sec. 7.1.2):
  mentioned only where container packaging touches chapter 17.

## Terminology used consistently here

- **TTFT** — time to first token, dominated by prefill. Report as
  percentiles (p50/p95/p99), never only as a mean.
- **TPS** — tokens per second. Per request, decode TPS is the reciprocal
  of per-token latency: `TPS ≈ 1 / TPOT` when TPOT is in seconds per
  token. System TPS aggregates across the batch and is the throughput
  figure batching raises. Say which one you mean.
- **TPOT** — time per output token after the first; the per-token
  latency whose reciprocal is per-request decode TPS.
- **Compute-bound / memory-bound** — whether arithmetic throughput or
  bytes moved per second sets the pace. Chapter 13 (book
  sec. 2.4) makes this quantitative with the ops:byte ratio; chapters
  00, 05, and 10 use the same words qualitatively until then.

See [GLOSSARY.md](GLOSSARY.md) for one-line definitions of every term
above plus ops:byte ratio, arithmetic intensity, ridge point, online
and offline inference, disaggregation, prefix caching, speculative
decoding, cold start, scale to zero, tensor and expert parallelism,
and MoE.
