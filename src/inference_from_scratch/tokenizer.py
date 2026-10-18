"""Ch 01 -- Tokenizers: text <-> token ids, three rungs of the ladder.

Why this exists in inference
-----------------------------
A model never sees characters or bytes; it sees integer ids in
``[0, vocab_size)`` and indexes an embedding table [V, D] with them.
The tokenizer is therefore the first and last stage of every inference
pipeline: ``prompt str -> encode -> [T] ids -> model -> [T'] ids ->
decode -> reply str``. A tokenizer bug does not crash -- it silently
changes the distribution the model was trained on, which is why
roundtrip losslessness is the property we test hardest.

The three rungs
---------------
1. ``ByteTokenizer`` -- id == utf-8 byte value (0..255). Zero training,
   lossless on *any* text by construction, but sequences are long
   (an emoji is 4 tokens).
2. ``CharTokenizer`` -- one id per distinct character in a corpus.
   Shorter than bytes for repeated non-ASCII, but *closed vocabulary*:
   an unseen character at inference time has no id at all.
3. ``BPETokenizer`` -- byte-level BPE. Starts from the 256 byte ids and
   *learns* merges for frequent adjacent pairs, so frequent substrings
   (" the", "ing", common emoji byte patterns) collapse to one id.
   This is the family Qwen / GPT use.

How production differs
-----------------------
Qwen ships a *pretrained* byte-level BPE with vocab_size ~151k, trained
on trillions of tokens with a regex pre-tokenizer (GPT-4-style pattern)
that splits text before BPE so merges never cross certain boundaries,
plus special tokens (<|endoftext|>, tool-call markers) that bypass
merging entirely. Our ``BPETokenizer.train`` implements the genuine
merge-learning loop (count pairs -> merge most frequent -> repeat) on
whatever small text you give it, with no pre-tokenizer and no special
tokens, so every step is inspectable. It is *not* a wrapper around
``tiktoken`` / ``tokenizers`` -- the counting loop below is the whole
algorithm those libraries optimise.
"""

from __future__ import annotations

from collections import Counter


# ---------------------------------------------------------------------------
# Rung 1: bytes
# ---------------------------------------------------------------------------
class ByteTokenizer:
    """Maps each utf-8 byte to the id equal to its byte value.

    encode: str -> list[int], each int in [0, 256), length == number of
            utf-8 bytes (NOT number of characters: 'é' is 2 ids).
    decode: list[int] -> str, exact inverse; lossless for any str,
            including emoji and combining characters, because utf-8
            itself is lossless.
    vocab_size is always 256.
    """

    vocab_size: int = 256

    def encode(self, text: str) -> list[int]:
        """str -> utf-8 bytes -> list of ints in [0, 256)."""
        # bytes objects iterate as ints in Python, so list() is the map.
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        """Inverse of encode. Raises UnicodeDecodeError if ids do not
        form valid utf-8 -- we deliberately do not hide that with
        errors='replace', because in a teaching tokenizer a corrupted
        byte stream is a bug to surface, not paper over. Note that a
        *truncated* multi-byte sequence (e.g. first half of an emoji)
        is exactly such a case."""
        return bytes(ids).decode("utf-8")


# ---------------------------------------------------------------------------
# Rung 2: characters
# ---------------------------------------------------------------------------
class CharTokenizer:
    """One id per distinct character seen at build time.

    Build with ``CharTokenizer.build(text)``. Ids are assigned in sorted
    character order so the mapping is deterministic across runs (set
    iteration order would not be).

    encode: str of length T -> list[int] of length T, ids in
            [0, vocab_size). Raises ValueError on a character that was
            not in the build text -- the closed-vocabulary failure mode
            that motivated subword tokenization.
    decode: exact inverse for in-vocabulary ids.
    """

    def __init__(self, chars: list[str]):
        # itos[i] is the character for id i; stoi is its inverse.
        self.itos: list[str] = list(chars)
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        """Collect the sorted set of characters in ``text`` as vocab."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """str -> list[int]; ValueError on unseen character."""
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as e:
            raise ValueError(
                f"character {e.args[0]!r} not in CharTokenizer vocabulary; "
                "CharTokenizer is closed-vocabulary (use Byte/BPE for open text)"
            ) from e

    def decode(self, ids: list[int]) -> str:
        """list[int] -> str by concatenating the char for each id."""
        return "".join(self.itos[i] for i in ids)


# ---------------------------------------------------------------------------
# Rung 3: byte-level BPE (genuinely trained merges)
# ---------------------------------------------------------------------------
class BPETokenizer:
    """Byte-level BPE with merges learned from a training text.

    Data structures (all plain, inspectable):
        merges: dict[(int, int), int] mapping an adjacent pair to the
                new token id created for it. Insertion order == learning
                order == merge rank (earliest merge = most frequent pair
                in the corpus at its step).
        vocab:  dict[int, bytes] mapping every token id to the raw byte
                string it expands to. Ids 0..255 are single bytes; merged
                ids are concatenations. decode is just "concatenate the
                bytes, utf-8 decode", so roundtrip losslessness reduces
                to "vocab expansions partition the original bytes".

    Use ``BPETokenizer.train(text, vocab_size)`` to build one.
    """

    def __init__(self, merges: dict[tuple[int, int], int], vocab: dict[int, bytes]):
        self.merges: dict[tuple[int, int], int] = dict(merges)
        self.vocab: dict[int, bytes] = dict(vocab)
        # Rank of each pair = order it was learned; encode applies
        # lowest-rank (earliest) merges first, mirroring training.
        self._ranks: dict[tuple[int, int], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

    # -- training -----------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        """Learn merges from ``text`` until vocab reaches ``vocab_size``.

        The genuine BPE loop (Sennrich et al. 2016, byte-level variant):
          1. Represent the corpus as a list of byte ids [N] (N = utf-8
             byte length of ``text``).
          2. Count every adjacent pair.
          3. Take the most frequent pair, mint the next id for it,
             record the merge, and replace every occurrence of the pair
             with the new id (the sequence shrinks).
          4. Repeat from 2 until we own ``vocab_size`` ids or no pair
             remains (corpus collapsed toward a single token).

        Ties are broken by first encounter in pair-count order; the
        result is deterministic for a given (text, vocab_size).

        Raises ValueError if vocab_size < 256 (bytes alone need 256).
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256 (byte base), got {vocab_size}")
        if not text:
            raise ValueError("cannot train BPE on empty text: no pairs to count")

        ids: list[int] = list(text.encode("utf-8"))
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        merges: dict[tuple[int, int], int] = {}

        num_merges = vocab_size - 256
        for step in range(num_merges):
            if len(ids) < 2:
                break  # whole corpus is one token; nothing left to merge
            # Step 2: count adjacent pairs over the current sequence.
            pair_counts = Counter(zip(ids, ids[1:]))
            # Step 3a: most frequent pair. Counter.most_common is stable
            # w.r.t. first-insertion order for ties on CPython dicts,
            # and zip order is corpus order, so this is deterministic.
            (a, b), _count = pair_counts.most_common(1)[0]
            new_id = 256 + step
            merges[(a, b)] = new_id
            vocab[new_id] = vocab[a] + vocab[b]
            # Step 3b: rewrite the sequence, replacing a,b -> new_id.
            ids = cls._apply_merge(ids, (a, b), new_id)
        return cls(merges, vocab)

    @staticmethod
    def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every non-overlapping occurrence of ``pair`` in ``ids``
        with ``new_id``, scanning left to right. Non-overlapping matters:
        merging (x, x) in [x, x, x] yields [new, x], not [new, new]."""
        out: list[int] = []
        i = 0
        a, b = pair
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # -- inference ----------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """str -> list[int] by applying learned merges in rank order.

        Start from byte ids, then repeatedly find the adjacent pair with
        the lowest merge rank (earliest learned = should be applied
        first) and merge all its occurrences, until no adjacent pair is
        a known merge. Worst case O(T * num_merges) for T bytes --
        production tokenizers use priority queues / regex chunking to
        make this near-linear; clarity wins here.
        Unseen text still encodes fine: unknown substrings simply stay
        as bytes. That open-vocabulary property is the point of BPE.
        """
        ids: list[int] = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # Find the lowest-rank mergeable pair present.
            best_pair: tuple[int, int] | None = None
            best_rank: int | None = None
            for pair in zip(ids, ids[1:]):
                rank = self._ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            ids = self._apply_merge(ids, best_pair, self.merges[best_pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """list[int] -> str: expand each id to its bytes, concatenate,
        utf-8 decode. Exact inverse of encode (encode only ever groups
        bytes, never alters them)."""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8")
