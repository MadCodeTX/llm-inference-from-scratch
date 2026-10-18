# 18 — Modalities: what changes when tokens are not text

> **Textbook framing** — Framing follows Kiely, *Inference Engineering* (2026), Ch 6. **Production question:** which parts of the text pipeline survive contact with images, audio, and embeddings, and which parts are replaced? **Status: read-only map.** No non-text code is built in this repo; every section below names the text mechanism it extends and marks future work as future. See [TEXTBOOK_MAP.md](TEXTBOOK_MAP.md).

## The one-sentence idea

Most modalities are the same decoder wearing a different front end — an encoder that turns pixels, waveforms, or text into vectors in the decoder's space — or, for embeddings, the same encoder with the decoder removed entirely.

## Why inference needs this

The chapters so far built one pipeline: tokenise, embed, attend, sample. Real deployments increasingly feed that pipeline things that are not typed text. The transferable skill is not any one modality's architecture; it is locating the seam where a new modality plugs in, and seeing which costs (prefill arithmetic, KV bytes, decode serialism) follow it through the seam.

## Vision-language models (VLM)

**Shape of the change.** An image is split into patches, run through a vision encoder, and projected into the language model's embedding dimension. The resulting image vectors are inserted into the token stream — typically as a run of embeddings occupying positions where text tokens would sit — and the ordinary decoder (chapters 02–07, unchanged) attends over the mixed sequence. Text tokens still arrive through the tokenizer and embedding table (chapters 01–02); image positions bypass the table and take the projector's output instead. A Qwen-VL-style deployment is the local pointer: the same Qwen decoder family this repo loads in chapter 09, fronted by a vision tower, served by the same engine layer as chapter 16.

**Inference consequences.** Image positions are prefill: they cost parallel arithmetic and, crucially, KV entries. A high-resolution image can contribute hundreds to thousands of sequence positions, so `kv_cache_bytes` (chapter 10) and TTFT (chapter 00) must be priced with those positions included — the "prompt" is longer than the visible text suggests. Decode is untouched: one token at a time, memory-bound, exactly as chapter 13 describes. Prefix caching (chapter 14) applies to the combined sequence, which is why repeated questions about one image benefit from caching its encoded positions.

## Embeddings

**Shape of the change.** An embedding model runs the encoder/decoder stack once over the input and pools the per-position outputs (mean pooling, or the final position's vector) into a single fixed-size representation. There is no sampling loop, no KV cache growth, and no decode phase at all.

**Inference consequences.** The workload is pure prefill-shaped: parallel, arithmetic-friendly, and batchable without the admission/retirement churn of chapter 11, because every request has a known, finite length and finishes in one forward. Throughput is therefore quoted in sequences or tokens per second at saturation — the offline pole of chapter 00 — and latency promises are correspondingly looser. This is the one modality here where this repo's machinery transfers almost literally: tokenise, forward, pool, with chapters 08 (sampling) and 05 (KV cache) simply not participating.

## Speech: ASR and TTS

**ASR (recognition)** is an encoder over audio frames (or spectrogram patches) feeding either a decoder that emits text tokens autoregressively, or a classification head over positions. The autoregressive variant inherits decode wholesale — TTFT, TPOT, KV cache, speculative options — with the audio encoding counted as an enlarged prefill. **TTS (synthesis)** runs the arrow backwards: text in, a decoder that emits audio tokens or spectral frames, often followed by a separate vocoder network that turns those into samples. Its serving profile is decode-like and streaming-sensitive, since callers play audio as it arrives; per-step cost includes the vocoder, which has no analogue in the text pipeline.

## Image and video generation

**Image generation** is usually not autoregressive over text tokens at all: an iterative denoising process refines a latent grid over a fixed number of steps, conditioned on a text encoding. The cost drivers rhyme with inference but do not map one-to-one — steps are known in advance (no EOS, no variable-length decode), each step is a large parallel forward (prefill-shaped), and there is no KV cache in the decoder sense. **Video generation** extends the same process over time as well as space, multiplying per-step arithmetic and memory by the frame count; it is the most capacity-hungry shape in this chapter and the furthest from anything this repo builds. Both are named here for orientation only — chapter 12's optimisation seams and chapter 13's roofline still help reason about them, but no mechanism transfers directly.

## The map, in one view

| Modality | Front end | Decoder reused? | Decode phase? | Dominant new cost |
|----------|-----------|-----------------|---------------|-------------------|
| VLM | Vision encoder + projector into embedding space | Yes, unchanged | Yes, text decode | Image positions in prefill and KV |
| Embeddings | Same text stack, pooled output | Encoder only | No | None — pure parallel forwards |
| ASR | Audio encoder | Usually, for text output | Yes, when autoregressive | Audio-length prefill |
| TTS | Text encoder | Autoregressive audio decoder + vocoder | Yes, audio tokens | Vocoder per step; streaming |
| Image gen | Text encoder as conditioner | No (denoising network) | Fixed-step iteration | Per-step latent forwards |
| Video gen | Text encoder as conditioner | No | Fixed-step iteration | Frames × per-step cost |

## Walk the code

There is deliberately no modality code to walk. The honest pointers are the seams a future build would touch: the embedding lookup in [model.py](../src/inference_from_scratch/model.py) (where image vectors would join the stream), `kv_cache_bytes` in [quantization.py](../src/inference_from_scratch/quantization.py) (where image positions get priced), and the pooled-forward shape implicit in [metrics.py](../src/inference_from_scratch/metrics.py) timings. Any future VLM or embeddings chapter must add its own code and tests before claiming more than this map.

## How the big systems do it

Production multimodal serving keeps the engine layer (chapter 16) and swaps encoders: requests carry typed inputs, encoders run as prefill-stage services (sometimes on separate pools, an echo of disaggregation), and the shared decoder sees one unified sequence. Embedding fleets are the simplest services in the building — stateless parallel forwards behind ordinary load balancing. Generative image/video systems run their own schedulers built around fixed step counts and large per-step memory, borrowing vocabulary (batching, percentiles, cost per output) rather than mechanisms from text serving.

## Common pitfalls

- **Pricing a VLM prompt by its text length.** Image positions dominate prefill and KV. Count positions, not characters.
- **Quoting embedding throughput in decode TPS.** There is no decode; the units are sequences/tokens per second in a prefill regime.
- **Assuming "multimodal" means "new decoder".** In VLMs the decoder is the product you already tuned; the encoder is the addition.
- **Letting a read-only map sound like a build.** This chapter ships no code. Roadmaps (see [ROADMAP.md](ROADMAP.md)) keep it labelled accordingly.

## Break it / exercises

1. A VLM request carries 12 text tokens and an image encoded as 576 positions. Using chapter 10's formula on the tiny config, price its KV at `B=1` in fp16 — then explain why the text-only estimate misleads.
2. Explain, from the loop in chapter 08, exactly which lines an embeddings service deletes and which it keeps.
3. For ASR and TTS, state which one can reuse chapter 14's speculative decoding unchanged and why the other needs re-examination.

## Check yourself

<details>
<summary>1. Where exactly do image representations enter a VLM's sequence?</summary>

At the embedding stage: projector outputs occupy sequence positions in place of token-embedding rows, and the unchanged decoder attends over the mixed stream. Those positions also occupy KV cache.
</details>

<details>
<summary>2. Why is embeddings serving the simplest production shape in this book?</summary>

One parallel forward per request, known length, pooled output, no decode loop, no KV growth, no sampling state — pure offline-style throughput work.
</details>

<details>
<summary>3. Which chapter-13 intuition survives in image generation, and which does not?</summary>

The roofline survives: per-step forwards are arithmetic/bandwidth priced as usual. The prefill/decode asymmetry does not: steps are fixed in number, with no autoregressive cache or variable-length serial phase.
</details>

## Summary in 30 seconds

New modalities mostly replace the front end, not the decoder — except embeddings, which delete the decoder's loop, and diffusion-style generation, which replaces the loop with fixed iterations. If you remember one pricing rule, make it "count image positions in prefill and KV"; if you remember one status line, make it that this chapter is a map, not a build.

## Where to go next

This completes the planned arc. Revisit [ROADMAP.md](ROADMAP.md) for the full curriculum status, or return to ch 13 and re-derive a bottleneck for a modality-shaped workload. Run `.venv/bin/python -m pytest -q` to confirm the text pipeline this map extends is still green.
