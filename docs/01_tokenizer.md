# 01 — Tokenizers: text becomes integers

## The one-sentence idea

A tokenizer is a lossless, reversible map between strings and integer ids in `[0, vocab_size)`, and the three rungs here — bytes, characters, learned BPE merges — trade sequence length against vocabulary openness.

## Why inference needs this

The model never sees text. It sees ids and indexes an embedding table `[V, D]` with them (chapter 02). The tokenizer is therefore the first and last stage of every request:

```text
prompt str --encode--> ids [T] --model--> new ids --decode--> reply str
```

A tokenizer bug rarely crashes. It silently feeds the model a different token stream than it was trained on, so the property worth testing hardest is **roundtrip losslessness**: `decode(encode(s)) == s`.

There is also a direct inference cost: longer sequences mean more prefill FLOPs and a bigger KV cache. A tokenizer that halves `T` roughly halves both.

## The math/shapes

Encoding changes length, not just representation:

| Tokenizer | Vocab | `"héllo"` (5 chars) | Length |
|-----------|-------|----------------------|--------|
| Byte | 256 | bytes of UTF-8: `h, é(2 bytes), l, l, o` | 6 ids |
| Char | corpus-dependent | one id per distinct char | 5 ids |
| BPE | 256 + merges | frequent pairs merged into single ids | ≤ 6, shrinks with training |

**Concrete example (bytes).** UTF-8 encodes `é` as two bytes, 195 and 169. So `ByteTokenizer().encode("é") == [195, 169]` — two tokens for one character. An emoji is typically 4 bytes, hence 4 tokens. This is the price of a fixed 256 vocab that can encode *anything*.

**Concrete example (BPE merges).** Train on `"aaab aaab"`. Byte ids start as `[97,97,97,98,32,97,97,97,98]`. The most frequent adjacent pair is `(97,97)` (4 occurrences), so merge #1 mints id 256 for `aa`. Applying it gives `[256,97,98,32,256,97,98]`. The next most frequent pair might be `(256,97)` → id 257 for `aaa`. Each merge shortens the corpus and grows the vocab by exactly one. Encoding new text applies merges in the order they were learned (lowest rank first).

## Walk the code

All three tokenizers live in [tokenizer.py](../src/inference_from_scratch/tokenizer.py) and share the interface `encode(str) -> list[int]`, `decode(list[int]) -> str`, `vocab_size`.

- `ByteTokenizer` (`tokenizer.py:48`): `encode` is `list(text.encode("utf-8"))` (`tokenizer.py:61`); `decode` is `bytes(ids).decode("utf-8")` (`tokenizer.py:66`). `vocab_size` is the class attribute `256` (`tokenizer.py:59`). Decode deliberately raises `UnicodeDecodeError` on a truncated multi-byte sequence instead of inserting replacement characters — a corrupt byte stream is a bug to surface.
- `CharTokenizer` (`tokenizer.py:79`): build with `CharTokenizer.build(text)` (`tokenizer.py:98`), which assigns ids in **sorted** character order for determinism. `encode` raises `ValueError` on any character absent from the build text (`tokenizer.py:107`) — the closed-vocabulary failure mode.
- `BPETokenizer` (`tokenizer.py:125`): `train(text, vocab_size)` (`tokenizer.py:152`) runs the genuine loop — count adjacent pairs with `Counter`, merge the most frequent, rewrite the sequence via `_apply_merge` (`tokenizer.py:197`), repeat. Two inspectable structures result: `merges: dict[(int,int), int]` and `vocab: dict[int, bytes]`. `encode` (`tokenizer.py:219`) repeatedly applies the lowest-rank merge present; `decode` (`tokenizer.py:246`) concatenates `vocab[i]` byte strings and UTF-8 decodes. `train` raises `ValueError` if `vocab_size < 256` or the text is empty.

Tests: [test_tokenizer.py](../tests/test_tokenizer.py) checks byte losslessness on non-ASCII text, char roundtrips, BPE roundtrips on its training alphabet, and that BPE encoding is no longer than byte encoding.

## How the big systems do it

**Qwen-family** models ship a *pretrained* byte-level BPE with a vocabulary on the order of 150k entries (read the exact `vocab_size` from the checkpoint's `config.json` — chapter 09 shows how). Two additions matter versus our teaching version: a regex **pre-tokenizer** splits text before BPE so merges never cross certain boundaries (digits, punctuation, whitespace), and **special tokens** (end-of-text, chat/tool markers) bypass merging entirely and are parsed atomically.

**vLLM-style serving** treats tokenization as a CPU stage ahead of the GPU batch: prompts are encoded with the model's own tokenizer (usually the fast Rust `tokenizers` library), and detokenization is streamed incrementally so a multi-byte character split across two sampled tokens is never decoded half-way — the exact `UnicodeDecodeError` case our `ByteTokenizer.decode` refuses to hide.

**llama.cpp / GGUF** stores the tokenizer (vocab + merges) inside the `.gguf` file itself, so the artifact is self-contained. The merge algorithm is the same counting loop as `BPETokenizer.train`, run once at model-build time over a huge corpus instead of at load time over a toy string.

Tradeoff summary: bigger vocab → shorter sequences (cheaper prefill, smaller KV cache) but a bigger embedding table `[V, D]` and a rarer-token training problem. Bytes are the floor everyone falls back to.

## A worked BPE trace

Corpus: `"low lower lowest"` is the classic textbook example; here is a shorter one you can verify against `BPETokenizer.train`. Corpus `"abab abab"` as bytes: `[97, 98, 97, 98, 32, 97, 98, 97, 98]`.

1. Pair counts: `(97, 98)` occurs 4 times; `(98, 97)` twice; `(98, 32)` and `(32, 97)` once each.
2. Merge 1: `(97, 98) -> 256` (`"ab"`). Rewrite: `[256, 256, 32, 256, 256]`. Length 9 → 5.
3. Pair counts now: `(256, 256)` occurs twice; `(256, 32)`, `(32, 256)` once. Merge 2: `(256, 256) -> 257` (`"abab"`). Rewrite: `[257, 32, 257]`. Length 5 → 3.
4. Encoding the *unseen* string `"abcab"`: bytes `[97, 98, 99, 97, 98]` → apply lowest-rank merge present, `(97, 98)`: `[256, 99, 256]`. The `c` (99) has no merges and simply survives as a byte — open vocabulary in action.
5. Decoding `[257, 32, 257]`: expand `vocab[257] = b"abab"`, `vocab[32] = b" "`, `vocab[257] = b"abab"` → `"abab abab"`. Lossless, as promised.

Two subtleties the code handles and readers should probe: merges never overlap (`_apply_merge` scans left-to-right, so merging `(x, x)` in `[x, x, x]` yields `[new, x]`, not `[new, new]`), and ties in pair counts break by first encounter, making training deterministic for a given corpus.

## Common pitfalls

- **Counting characters instead of bytes.** `len("é") == 1` but `ByteTokenizer` emits 2 ids. Prompt-length limits, KV-cache sizing, and pricing all count *tokens*, i.e. ids, not characters.
- **Assuming token boundaries respect words.** BPE merges are purely statistical; a token can span half a word or include a leading space (`" the"` is famously one token in GPT-style vocabs). Detokenization must concatenate everything and decode once.
- **Decoding partial streams.** In streaming serving, a multi-byte character can be split across two sampled tokens. Decode incrementally with a byte buffer; decoding each token separately raises exactly the `UnicodeDecodeError` that `ByteTokenizer.decode` surfaces on truncated input.

## Break it / exercises

1. Encode `"café ☕"` with `ByteTokenizer` and count the ids. Decode `[195]` alone and observe the `UnicodeDecodeError`. When could a sampler hand you exactly that truncated list?
2. Build a `CharTokenizer` on `"hello"` and encode `"world"`. Which `ValueError` fires and why can BPE never fire it?
3. Train `BPETokenizer` on a repetitive string (e.g. `"the the the the"`) with `vocab_size=260`. Print `merges` and verify the first merge is the most frequent byte pair. Then encode unseen text containing none of the merges and confirm it degrades to plain bytes.

## Check yourself

<details>
<summary>1. Why is byte-level BPE called "open vocabulary" while the char tokenizer is "closed"?</summary>

Because BPE starts from all 256 bytes: any string encodes to *something*, with unseen substrings simply staying as bytes. The char tokenizer has no id at all for an unseen character and must raise.
</details>

<details>
<summary>2. <code>decode(encode(s)) == s</code> holds for all three tokenizers. What second property makes BPE useful anyway?</summary>

Compression: frequent substrings collapse to one id, shortening `T`, which cuts prefill cost and KV-cache size. Losslessness alone is satisfied by bytes.
</details>

<details>
<summary>3. In what order does <code>BPETokenizer.encode</code> apply merges, and why?</summary>

Lowest rank (earliest learned) first, mirroring training. Applying a later merge first could consume bytes that an earlier, more frequent merge should have claimed.
</details>

## Summary in 30 seconds

This chapter's core is lossless text-to-id maps and the length-for-vocab trade. If you remember one formula, make it the shape or cost expression derived in *The math/shapes*; if you remember one code location, make it the first entry in *Walk the code*. Everything else — production variants, pitfalls, exercises — is commentary on those two anchors.

Re-derive, don't re-read: close the book and reproduce the tiny numeric example by hand, then run the matching test file and confirm the machine agrees. Disagreement between your hand calculation and the test is the highest-value bug you can find at this stage, whether it lives in the code or in your understanding.

## Where to go next

Continue to ch 02 (embeddings) looks each id up in a [V, D] table. If a concept from an earlier chapter felt shaky here, revisit it now rather than pushing on: this curriculum is strictly cumulative, and later chapters (especially 09 and 11) assume the vocabulary of every chapter before them. The [glossary](GLOSSARY.md) gives one-line refreshers for prefill, decode, KV cache, GQA, RoPE, logits, TTFT, TPOT, quantization, GGUF, safetensors, and paged attention.

### Reader checkpoints

Before moving on, you should be able to do each of these without looking back:

- State the chapter's one-sentence idea in your own words, including *why* inference (not training) forces the design.
- Reproduce the tiny numeric example with pencil arithmetic and match every digit the chapter quotes.
- Name the primary source file and function for the chapter and describe its inputs, outputs, and shapes from memory.
- Explain the grouped-query / bandwidth / scheduling consequence (whichever this chapter carries) in terms of the byte or shape formulas, not adjectives.
- Predict which existing test breaks first under each *Break it* mutation, and why the failure is loud or silent.

Run the chapter's test file now: `.venv/bin/python -m pytest tests/test_tokenizer.py -q`. Every test named in *Walk the code* should pass unmodified on this machine; if one does not, you have found either an environment problem (check `.venv` and the torch CPU wheel) or a genuine regression — both worth resolving before the next chapter, because later chapters stack on these exact functions.

A final intuition: tokenization is the only chapter whose choices are *frozen into the checkpoint*. Weights can be re-quantized and caches re-laid-out, but a model's embedding rows are indexed by token id, so the tokenizer that trained it is part of its identity. That is why GGUF embeds the tokenizer in the file (ch 09) and why speculative decoding requires draft and target to share one (ch 12).
