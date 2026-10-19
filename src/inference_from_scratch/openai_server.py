"""OpenAI-compatible serving surface (stdlib ``http.server`` only).

Smallest production-shaped HTTP layer needed for
``benchmarks/run_bench.py`` (and ``benchmarks/quality_probe.py``) to
measure this engine in the same harness as vLLM:

* ``POST /v1/completions`` -- streaming (SSE, one chunk per token,
  final chunk carries ``usage`` when ``stream_options.include_usage``)
  and non-streaming. Backends that support sampling accept
  ``temperature``/``top_p``/``top_k``/``seed``; greedy-only backends
  reject ``temperature > 0`` with 400 so exactness is never silently
  violated.
* ``GET /health`` and ``GET /v1/models`` -- liveness / model list.

Backend seam (injection for CPU tests, no 27B weights needed):

* A backend provides ``model_id``, ``eos_id``, ``encode(text)``,
  ``decode(ids)`` and ``stream_token_ids(prompt_ids, max_tokens,
  cancelled, ignore_eos=False, sampling=None) -> Iterator[int]``
  (raw ids, up to ``max_tokens``; the server owns EOS/stop/
  max_tokens/usage/finish_reason policy). ``sampling`` is a dict with
  ``temperature``/``top_p``/``top_k``/``seed`` keys.
* :class:`FakeBackend` -- deterministic char-level fake.
* :class:`BatchedEngineBackend` -- drives a ``TPEngineCore``-shaped
  engine (``submit``/``poll``/``step``/``has_work``) from one driver
  thread, so concurrent HTTP requests batch in the engine. An
  optional ``sync_hook`` broadcasts submissions/steps to TP worker
  ranks (see ``scripts/serve_qwen38_tp2.py``); only the leader/rank 0
  constructs the HTTP server, workers run :func:`participate`.

Cancellation/timeout: a client disconnect or the server request
timeout sets the ``cancelled`` event; backends check it between
tokens. A cancelled engine request cannot be preempted mid-step --
its remaining tokens are discarded by the driver (bounded by
``max_tokens``), never delivered.
"""
from __future__ import annotations

import json
import queue
import random
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator, Protocol

__all__ = ["FakeBackend", "BatchedEngineBackend", "HFBackend",
           "make_server", "serve_forever", "participate",
           "encode_lockstep_command", "decode_lockstep_command",
           "participate_compact"]


class Backend(Protocol):
    model_id: str
    eos_id: int | None
    greedy_only: bool

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def stream_token_ids(self, prompt_ids: list[int], max_tokens: int,
                         cancelled: threading.Event,
                         ignore_eos: bool = False,
                         sampling: dict | None = None) -> Iterator[int]: ...


def _sampling_kwargs(sampling: dict | None) -> dict:
    """Engine-submit kwargs from a server sampling dict (greedy default)."""
    if not sampling:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": None, "seed": None}
    return {"temperature": float(sampling.get("temperature") or 0.0),
            "top_p": float(sampling.get("top_p", 1.0)),
            "top_k": sampling.get("top_k"),
            "seed": sampling.get("seed")}


def _is_sampled(sampling: dict | None) -> bool:
    return bool(sampling) and float(sampling.get("temperature") or 0.0) != 0.0


# --------------------------------------------------------------------------
# Fake backend (CPU tests / smoke)
# --------------------------------------------------------------------------
class FakeBackend:
    """Deterministic char-token fake: id == unicode code point.

    Generation cycles a fixed word list as characters, seeded only by
    prompt length, so concurrent requests are isolated and repeatable.
    ``scripted_ids`` forces an exact id sequence (EOS/stop tests).
    ``token_delay_s`` simulates a slow backend (timeout tests).
    """

    greedy_only = False
    _WORDS = " alpha beta gamma delta epsilon zeta eta theta"

    def __init__(self, model_id: str = "fake-model", eos_id: int | None = None,
                 scripted_ids: list[int] | None = None, token_delay_s: float = 0.0):
        self.model_id = model_id
        self.eos_id = eos_id
        self._scripted = list(scripted_ids) if scripted_ids is not None else None
        self.token_delay_s = float(token_delay_s)

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(i)) for i in ids if int(i) != (self.eos_id or -1))

    def stream_token_ids(self, prompt_ids, max_tokens, cancelled, ignore_eos=False,
                         sampling=None):
        if self._scripted is not None:
            ids = self._scripted[:max_tokens]
        elif _is_sampled(sampling):
            # Seeded fake sampling: reproducible per (seed, prompt),
            # so sampled smoke tests are deterministic without weights.
            seed = sampling.get("seed") if sampling else None
            rng = random.Random(
                (int(seed) if seed is not None else 0) * 1000003
                + len(list(prompt_ids)))
            pool = [ord(c) for c in self._WORDS]
            ids = [rng.choice(pool) for _ in range(int(max_tokens))]
        else:
            text = (self._WORDS * (max_tokens // len(self._WORDS) + 2))
            ids = [ord(c) for c in text[:max_tokens]]
        for tid in ids:
            if cancelled.is_set():
                return
            if self.token_delay_s:
                if cancelled.wait(self.token_delay_s):
                    return
            yield int(tid)


# --------------------------------------------------------------------------
# HF tokenizer backend wrapper (real tokenizer, injected generator fn)
# --------------------------------------------------------------------------
class HFBackend:
    """Wraps an HF tokenizer + a ``generate_ids`` callable.

    ``generate_ids(prompt_ids, max_tokens, cancelled)`` yields raw ids.
    Used by the TP script so tokenizer behaviour is the real Qwen one
    while generation stays behind the engine seam.
    """

    greedy_only = True

    def __init__(self, model_id: str, tokenizer, generate_ids: Callable,
                 eos_id: int | None = None):
        self.model_id = model_id
        self._tok = tokenizer
        self._gen = generate_ids
        if eos_id is None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
        self.eos_id = None if eos_id is None else int(eos_id)

    def encode(self, text: str) -> list[int]:
        try:
            ids = self._tok.encode(text, add_special_tokens=False)
        except TypeError:
            ids = self._tok.encode(text)
        return [int(t) for t in ids]

    def decode(self, ids: list[int]) -> str:
        ids = [int(t) for t in ids]
        try:
            return self._tok.decode(ids, skip_special_tokens=False)
        except TypeError:
            return self._tok.decode(ids)

    def stream_token_ids(self, prompt_ids, max_tokens, cancelled, ignore_eos=False,
                         sampling=None):
        # The injected generator yields raw ids; EOS policy stays in the
        # server layer, so ignore_eos needs no change here. Greedy only:
        # the server rejects temperature>0 for this backend.
        yield from self._gen(list(prompt_ids), int(max_tokens), cancelled)


# --------------------------------------------------------------------------
# Batched TPEngineCore backend (single-process, or leader side of TP)
# --------------------------------------------------------------------------
class _Job:
    __slots__ = ("prompt_ids", "max_tokens", "cancelled", "tokens", "done", "error",
                 "ignore_eos", "sampling")

    def __init__(self, prompt_ids, max_tokens, cancelled, ignore_eos=False,
                 sampling=None):
        self.prompt_ids, self.max_tokens, self.cancelled = prompt_ids, max_tokens, cancelled
        self.ignore_eos = bool(ignore_eos)
        self.sampling = sampling
        self.tokens: queue.Queue = queue.Queue()
        self.done = threading.Event()
        self.error: BaseException | None = None


class BatchedEngineBackend:
    """One driver thread steps the engine; HTTP threads read queues.

    ``sync_hook(submits, do_step)`` (TP leader only) is called by the
    driver before each engine step with newly queued
    ``(prompt_ids, max_tokens)`` pairs, so workers submit/step in
    lockstep. Without it the backend is single-process.
    """

    greedy_only = False

    def __init__(self, model_id: str, engine, encode, decode,
                 eos_id: int | None = None, sync_hook: Callable | None = None):
        self.model_id, self.engine = model_id, engine
        self._encode, self._decode = encode, decode
        self.eos_id = eos_id if eos_id is not None else getattr(engine, "eos_id", None)
        self._sync_hook = sync_hook
        self._pending: queue.Queue[_Job] = queue.Queue()
        self._active: dict[int, _Job] = {}
        self._seen: dict[int, int] = {}
        self._driver: threading.Thread | None = None
        self._stop = threading.Event()
        self._closed = False
        self._close_lock = threading.Lock()

    def encode(self, text): return self._encode(text)
    def decode(self, ids): return self._decode(ids)

    def _ensure_driver(self):
        if self._closed:
            raise RuntimeError("backend closed")
        if self._driver is None or not self._driver.is_alive():
            self._stop.clear()
            self._driver = threading.Thread(target=self._drive, daemon=True,
                                            name="engine-driver")
            self._driver.start()

    def _fail(self, jobs, exc):
        for job in jobs:
            if job.error is None:
                job.error = exc
            job.done.set()

    def _drive(self):
        pending_submits: list[tuple] = []
        while not self._stop.is_set():
            # Poll with a timeout even when idle so close() wakes us.
            try:
                job = self._pending.get(timeout=0.05)
                jobs = [job]
            except queue.Empty:
                jobs = []
            while True:
                try:
                    jobs.append(self._pending.get_nowait())
                except queue.Empty:
                    break
            for job in jobs:
                if self._stop.is_set():
                    self._fail([job], RuntimeError("backend closed"))
                    continue
                try:
                    rid = self.engine.submit(job.prompt_ids, job.max_tokens,
                                             ignore_eos=job.ignore_eos,
                                             **_sampling_kwargs(job.sampling))
                except Exception as exc:  # e.g. prompt+max_tokens > max_seq_len
                    self._fail([job], exc)
                    continue
                self._active[rid] = job
                self._seen[rid] = 0
                pending_submits.append((job.prompt_ids, job.max_tokens,
                                        job.ignore_eos, job.sampling))
            if self._stop.is_set():
                self._fail(list(self._active.values()),
                           RuntimeError("backend closed"))
                self._active.clear(); self._seen.clear()
                pending_submits = []
                break
            if not self._active:
                pending_submits = []
                continue
            try:
                if self._sync_hook is not None:
                    self._sync_hook(pending_submits, True)
                pending_submits = []
                self.engine.step()
                for rid, job in list(self._active.items()):
                    snap = self.engine.poll(rid)
                    ids = snap["token_ids"]
                    for tid in ids[self._seen[rid]:]:
                        if not job.cancelled.is_set():
                            job.tokens.put(int(tid))
                    self._seen[rid] = len(ids)
                    if snap["status"] == "finished":
                        job.done.set()
                        del self._active[rid], self._seen[rid]
            except Exception as exc:
                # Engine state is uncertain after a step/poll failure:
                # fail every active job instead of hanging its consumer.
                self._fail(list(self._active.values()), exc)
                self._active.clear(); self._seen.clear()
                pending_submits = []

    def stream_token_ids(self, prompt_ids, max_tokens, cancelled, ignore_eos=False,
                         sampling=None):
        self._ensure_driver()
        job = _Job(list(prompt_ids), int(max_tokens), cancelled,
                   ignore_eos=ignore_eos, sampling=sampling)
        self._pending.put(job)
        while True:
            if job.error is not None:
                raise job.error
            if cancelled.is_set() and job.done.is_set():
                return
            try:
                tid = job.tokens.get(timeout=0.05)
            except queue.Empty:
                if job.error is not None:
                    raise job.error
                if job.done.is_set():
                    while not job.tokens.empty():
                        yield job.tokens.get_nowait()
                    if job.error is not None:
                        raise job.error
                    return
                if cancelled.is_set():
                    return
                continue
            yield tid

    def close(self):
        """Idempotent: fail in-flight/pending jobs, stop driver, then shutdown.

        The shutdown broadcast is sent only after the driver thread has
        stopped (or timed out), so it cannot interleave with a step
        broadcast from the driver. A second close() is a no-op.
        """
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        driver = self._driver
        if driver is not None and driver.is_alive():
            driver.join(timeout=5.0)
        # Fail anything the driver did not finish (in-flight or queued),
        # so HTTP consumers error instead of hanging.
        self._fail(list(self._active.values()), RuntimeError("backend closed"))
        self._active.clear(); self._seen.clear()
        while True:
            try:
                self._fail([self._pending.get_nowait()],
                           RuntimeError("backend closed"))
            except queue.Empty:
                break
        if self._sync_hook is not None:
            try:
                self._sync_hook([], False, shutdown=True)
            except Exception:
                pass


def participate(engine, recv: Callable, eos_id: int | None = None):
    """TP worker loop: mirror leader submissions/steps, never serve HTTP.

    ``recv()`` returns ``(submits, do_step, shutdown)`` (blocking
    collective, e.g. ``dist.broadcast_object_list`` from rank 0).
    """
    while True:
        submits, do_step, shutdown = recv()
        if shutdown:
            return
        for item in submits:
            prompt_ids, max_new = item[0], item[1]
            ignore_eos = bool(item[2]) if len(item) > 2 else False
            sampling = item[3] if len(item) > 3 else None
            engine.submit(list(prompt_ids), int(max_new), ignore_eos=ignore_eos,
                          **_sampling_kwargs(sampling))
        if do_step:
            engine.step()


def _float_to_ints(x: float) -> list[int]:
    import struct
    bits = struct.unpack("<Q", struct.pack("<d", float(x)))[0]
    hi, lo = bits >> 32, bits & 0xFFFFFFFF
    return [hi - (1 << 32) if hi >= (1 << 31) else hi,
            lo - (1 << 32) if lo >= (1 << 31) else lo]


def _ints_to_float(hi: int, lo: int) -> float:
    import struct
    bits = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _encode_sampling(sampling) -> list[int]:
    kw = _sampling_kwargs(sampling)
    return [*_float_to_ints(kw["temperature"]),
            *_float_to_ints(kw["top_p"]),
            int(kw["top_k"]) if kw["top_k"] is not None else 0,
            int(kw["seed"]) if kw["seed"] is not None else -1]


def _decode_sampling(vals) -> dict:
    return {"temperature": _ints_to_float(vals[0], vals[1]),
            "top_p": _ints_to_float(vals[2], vals[3]),
            "top_k": int(vals[4]) if vals[4] else None,
            "seed": int(vals[5]) if vals[5] >= 0 else None}


def encode_lockstep_command(submits, do_step: bool,
                            shutdown: bool = False) -> list[int]:
    """Pack one lockstep command as flat ints (no pickle).

    Layout: ``[shutdown, do_step, n_submits, (prompt_len, max_tokens,
    ignore_eos, *prompt_ids) * n_submits]``. A step-only command is
    3 ints, so the TP script can carry it in one small tensor
    broadcast instead of ``broadcast_object_list`` per engine step.
    When any submit carries a sampling dict (4th tuple element), a
    trailing block ``[n_submits, (temp_hi, temp_lo, topp_hi, topp_lo,
    top_k, seed) * n_submits]`` is appended; temperature/top_p are
    carried as exact float64 bit patterns (two signed 32-bit ints
    each), top_k 0=None, seed -1=None. Greedy 3-tuples encode
    byte-identically to before.
    """
    submits = list(submits)
    out = [1 if shutdown else 0, 1 if do_step else 0, len(submits)]
    for item in submits:
        prompt_ids, max_new, *rest = item
        ids = [int(t) for t in prompt_ids]
        out += [len(ids), int(max_new), 1 if (rest and rest[0]) else 0, *ids]
    if any(len(item) > 3 and item[3] is not None for item in submits):
        out.append(len(submits))
        for item in submits:
            sampling = item[3] if len(item) > 3 else None
            out += _encode_sampling(sampling)
    return out


def decode_lockstep_command(payload) -> tuple[list, bool, bool]:
    """Inverse of :func:`encode_lockstep_command`; raises on truncation."""
    vals = [int(v) for v in payload]
    if len(vals) < 3:
        raise ValueError(f"lockstep command too short: {len(vals)} ints")
    shutdown, do_step, n = bool(vals[0]), bool(vals[1]), vals[2]
    if n < 0:
        raise ValueError(f"lockstep command has negative submit count {n}")
    submits, i = [], 3
    for _ in range(n):
        if i + 3 > len(vals):
            raise ValueError("truncated lockstep submit header")
        plen, max_new, ignore_eos = vals[i], vals[i + 1], bool(vals[i + 2])
        i += 3
        if plen < 0 or i + plen > len(vals):
            raise ValueError("truncated lockstep prompt ids")
        submits.append((vals[i:i + plen], int(max_new), ignore_eos))
        i += plen
    if i < len(vals):  # trailing sampling block (see encode)
        if vals[i] != n or i + 1 + 6 * n > len(vals):
            raise ValueError("truncated lockstep sampling block")
        i += 1
        sampled = []
        for sub in submits:
            sampled.append((*sub, _decode_sampling(vals[i:i + 6])))
            i += 6
        submits = sampled
    return submits, do_step, shutdown


def participate_compact(engine, recv_command: Callable):
    """Like :func:`participate`, but ``recv_command()`` returns flat ints.

    Errors from ``engine.submit``/``engine.step`` propagate to the
    caller (the TP script destroys the process group in ``finally``),
    matching the object protocol's failure behaviour.
    """
    while True:
        submits, do_step, shutdown = decode_lockstep_command(recv_command())
        if shutdown:
            return
        for item in submits:
            prompt_ids, max_new, ignore_eos = item[0], item[1], item[2]
            sampling = item[3] if len(item) > 3 else None
            engine.submit(list(prompt_ids), int(max_new),
                          ignore_eos=bool(ignore_eos),
                          **_sampling_kwargs(sampling))
        if do_step:
            engine.step()


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------
def _err(message: str, type_: str = "invalid_request_error",
         param: str | None = None, code: str | None = None) -> dict:
    return {"error": {"message": message, "type": type_,
                      "param": param, "code": code}}


class _State:
    def __init__(self, backend, request_timeout_s: float | None):
        self.backend = backend
        self.request_timeout_s = request_timeout_s
        self.created = int(time.time())


def _make_handler(state: _State):
    backend = state.backend

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "inference-from-scratch/0.1"

        def log_message(self, *a):  # keep test/bench output clean
            pass

        # -- helpers --------------------------------------------------
        def _send_json(self, status: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            return json.loads(raw.decode("utf-8"))

        # -- routes ---------------------------------------------------
        def do_GET(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/health":
                self._send_json(200, {"status": "ok", "model": backend.model_id})
            elif path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [{
                    "id": backend.model_id, "object": "model",
                    "created": state.created, "owned_by": "inference-from-scratch"}]})
            elif path == "/v1/models/" + backend.model_id:
                self._send_json(200, {"id": backend.model_id, "object": "model",
                                      "created": state.created,
                                      "owned_by": "inference-from-scratch"})
            else:
                self._send_json(404, _err(f"unknown route {path}",
                                          type_="not_found_error", code="not_found"))

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path != "/v1/completions":
                self._send_json(404, _err(f"unknown route {path}",
                                          type_="not_found_error", code="not_found"))
                return
            try:
                req = self._read_json()
            except Exception as exc:
                self._send_json(400, _err(f"invalid JSON body: {exc}", param="body"))
                return
            self._completions(req)

        # -- completions ------------------------------------------------
        def _completions(self, req: dict):
            if not isinstance(req, dict):
                self._send_json(400, _err("request body must be a JSON object"))
                return
            model = req.get("model") or backend.model_id
            if model != backend.model_id:
                self._send_json(404, _err(f"model {model!r} not found",
                                          type_="not_found_error", param="model",
                                          code="model_not_found"))
                return
            if req.get("n", 1) not in (1, None):
                self._send_json(400, _err("only n=1 is supported", param="n"))
                return
            temp = req.get("temperature", 0.0)
            try:
                temp = 0.0 if temp is None else float(temp)
            except (TypeError, ValueError):
                self._send_json(400, _err("temperature must be a number", param="temperature"))
                return
            if getattr(backend, "greedy_only", True) and temp != 0.0:
                self._send_json(400, _err(
                    "only greedy decoding (temperature=0) is supported; "
                    "sampled arms require a sampling backend", param="temperature"))
                return
            top_p_raw = req.get("top_p", 1.0)
            try:
                top_p = 1.0 if top_p_raw is None else float(top_p_raw)
            except (TypeError, ValueError):
                self._send_json(400, _err("top_p must be a number", param="top_p"))
                return
            if not (0.0 < top_p <= 1.0):
                self._send_json(400, _err("top_p must be in (0, 1]", param="top_p"))
                return
            top_k = req.get("top_k")
            if top_k is not None and (isinstance(top_k, bool)
                                      or not isinstance(top_k, int) or top_k < 1):
                self._send_json(400, _err("top_k must be an integer >= 1",
                                          param="top_k"))
                return
            seed = req.get("seed")
            if seed is not None and (isinstance(seed, bool)
                                     or not isinstance(seed, int)
                                     or not (0 <= seed < 2 ** 63)):
                self._send_json(400, _err("seed must be a non-negative integer",
                                          param="seed"))
                return
            sampling = {"temperature": temp, "top_p": top_p,
                        "top_k": top_k, "seed": seed}
            # prompt: string, or explicit token-id list.
            prompt = req.get("prompt")
            if isinstance(prompt, str):
                if prompt == "":
                    self._send_json(400, _err("prompt must be non-empty", param="prompt"))
                    return
                try:
                    prompt_ids = backend.encode(prompt)
                except Exception as exc:
                    self._send_json(400, _err(f"tokenization failed: {exc}", param="prompt"))
                    return
            elif isinstance(prompt, list) and prompt and all(
                    isinstance(t, int) and not isinstance(t, bool) for t in prompt):
                prompt_ids = [int(t) for t in prompt]
            else:
                self._send_json(400, _err(
                    "prompt must be a non-empty string or list of token ids",
                    param="prompt"))
                return
            max_tokens = req.get("max_tokens", 16)
            if max_tokens is None:
                max_tokens = 16
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
                self._send_json(400, _err("max_tokens must be a non-negative integer",
                                          param="max_tokens"))
                return
            stop = req.get("stop")
            if isinstance(stop, str):
                stop = [stop]
            if stop is not None and not (isinstance(stop, list)
                                         and all(isinstance(s, str) for s in stop)):
                self._send_json(400, _err("stop must be a string or list of strings",
                                          param="stop"))
                return
            ignore_eos = bool(req.get("ignore_eos", False))
            stream = bool(req.get("stream", False))
            include_usage = bool((req.get("stream_options") or {}).get("include_usage"))

            cancelled = threading.Event()
            deadline = (time.monotonic() + state.request_timeout_s
                        if state.request_timeout_s else None)
            cid = "cmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            gen_ids: list[int] = []
            text_so_far = ""
            finish_reason: str | None = None

            def gen():
                """Yield (delta_text, token_id); sets finish_reason."""
                nonlocal text_so_far, finish_reason
                if max_tokens == 0:
                    finish_reason = "length"
                    return
                try:
                    it = backend.stream_token_ids(list(prompt_ids), max_tokens, cancelled,
                                                  ignore_eos=ignore_eos,
                                                  sampling=sampling)
                    for tid in it:
                        if cancelled.is_set():
                            finish_reason = "cancelled"
                            return
                        if deadline and time.monotonic() > deadline:
                            cancelled.set()
                            finish_reason = "timeout"
                            return
                        tid = int(tid)
                        if (not ignore_eos and backend.eos_id is not None
                                and tid == backend.eos_id):
                            finish_reason = "stop"
                            return  # EOS text is never emitted
                        gen_ids.append(tid)
                        prev_text = text_so_far
                        new_text = backend.decode(list(gen_ids))
                        delta = (new_text[len(prev_text):]
                                 if new_text.startswith(prev_text) else new_text)
                        text_so_far = new_text
                        # stop sequences: truncate at first occurrence
                        hit = None
                        for s in (stop or []):
                            if s and s in text_so_far:
                                idx = text_so_far.index(s)
                                if hit is None or idx < hit[0]:
                                    hit = (idx, s)
                        if hit is not None:
                            keep = text_so_far[:hit[0]]
                            delta = (keep[len(prev_text):]
                                     if keep.startswith(prev_text) else keep)
                            text_so_far = keep
                            finish_reason = "stop"
                            if delta:
                                yield delta, tid
                            return
                        if delta:
                            yield delta, tid
                        if len(gen_ids) >= max_tokens:
                            finish_reason = "length"
                            return
                except Exception:
                    raise
                if finish_reason is None:
                    finish_reason = "length" if len(gen_ids) >= max_tokens else "stop"

            usage = lambda: {"prompt_tokens": len(prompt_ids),  # noqa: E731
                             "completion_tokens": len(gen_ids),
                             "total_tokens": len(prompt_ids) + len(gen_ids)}

            def chunk(delta: str, fr, with_usage=False):
                if fr not in (None, "stop", "length"):
                    fr = "stop"  # cancelled/timeout are not OpenAI finish reasons
                c = {"id": cid, "object": "text_completion", "created": created,
                     "model": backend.model_id,
                     "choices": [{"text": delta, "index": 0,
                                  "logprobs": None, "finish_reason": fr}]}
                if with_usage:
                    c["usage"] = usage()
                return c

            if not stream:
                try:
                    for _d, _t in gen():
                        pass
                except Exception as exc:
                    self._send_json(500, _err(f"generation failed: {exc}",
                                              type_="server_error", code="generation_failed"))
                    return
                if finish_reason in ("timeout",):
                    self._send_json(504, _err("request timed out",
                                              type_="server_error", code="timeout"))
                    return
                body = chunk(text_so_far, finish_reason or "stop")
                body["usage"] = usage()
                self._send_json(200, body)
                return

            # Streaming SSE. Headers sent lazily on first chunk so a
            # pre-first-token failure can still be an HTTP error.
            sent_headers = False
            try:
                for delta, _t in gen():
                    if not sent_headers:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.close_connection = True
                        self.end_headers()
                        sent_headers = True
                    payload = "data: " + json.dumps(chunk(delta, None)) + "\n\n"
                    self.wfile.write(payload.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                cancelled.set()
                return
            except Exception as exc:
                if not sent_headers:
                    self._send_json(500, _err(f"generation failed: {exc}",
                                              type_="server_error", code="generation_failed"))
                else:
                    try:
                        self.wfile.write(("data: " + json.dumps(
                            _err(f"generation failed: {exc}", type_="server_error")) + "\n\n").encode())
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except Exception:
                        cancelled.set()
                return
            if finish_reason == "timeout" and not sent_headers:
                self._send_json(504, _err("request timed out",
                                          type_="server_error", code="timeout"))
                return
            if not sent_headers:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
            try:
                final = chunk("", finish_reason or "stop", with_usage=include_usage)
                self.wfile.write(("data: " + json.dumps(final) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                cancelled.set()

    return Handler


def make_server(backend, host: str = "127.0.0.1", port: int = 8000,
                request_timeout_s: float | None = 600.0) -> ThreadingHTTPServer:
    """Build (but do not start) the HTTP server for ``backend``."""
    state = _State(backend, request_timeout_s)
    srv = ThreadingHTTPServer((host, int(port)), _make_handler(state))
    srv.daemon_threads = True
    return srv


def serve_forever(backend, host: str = "127.0.0.1", port: int = 8000,
                  request_timeout_s: float | None = 600.0):
    srv = make_server(backend, host, port, request_timeout_s)
    print(f"serving {backend.model_id} on http://{host}:{srv.server_address[1]}",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        close = getattr(backend, "close", None)
        if callable(close):
            close()

