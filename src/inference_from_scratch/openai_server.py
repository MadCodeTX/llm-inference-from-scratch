"""OpenAI-compatible serving surface (stdlib ``http.server`` only).

Smallest production-shaped HTTP layer needed for
``benchmarks/run_bench.py`` (and ``benchmarks/quality_probe.py``) to
measure this engine in the same harness as vLLM:

* ``POST /v1/completions`` -- streaming (SSE, one chunk per token,
  final chunk carries ``usage`` when ``stream_options.include_usage``)
  and non-streaming. Greedy only: ``temperature > 0`` is rejected so
  exactness is never silently violated.
* ``GET /health`` and ``GET /v1/models`` -- liveness / model list.

Backend seam (injection for CPU tests, no 27B weights needed):

* A backend provides ``model_id``, ``eos_id``, ``encode(text)``,
  ``decode(ids)`` and ``stream_token_ids(prompt_ids, max_tokens,
  cancelled) -> Iterator[int]`` (raw ids, up to ``max_tokens``; the
  server owns EOS/stop/max_tokens/usage/finish_reason policy).
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
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator, Protocol

__all__ = ["FakeBackend", "BatchedEngineBackend", "HFBackend",
           "make_server", "serve_forever", "participate"]


class Backend(Protocol):
    model_id: str
    eos_id: int | None
    greedy_only: bool

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def stream_token_ids(self, prompt_ids: list[int], max_tokens: int,
                         cancelled: threading.Event,
                         ignore_eos: bool = False) -> Iterator[int]: ...


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

    greedy_only = True
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

    def stream_token_ids(self, prompt_ids, max_tokens, cancelled, ignore_eos=False):
        if self._scripted is not None:
            ids = self._scripted[:max_tokens]
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

    def stream_token_ids(self, prompt_ids, max_tokens, cancelled, ignore_eos=False):
        # The injected generator yields raw ids; EOS policy stays in the
        # server layer, so ignore_eos needs no change here.
        yield from self._gen(list(prompt_ids), int(max_tokens), cancelled)


# --------------------------------------------------------------------------
# Batched TPEngineCore backend (single-process, or leader side of TP)
# --------------------------------------------------------------------------
class _Job:
    __slots__ = ("prompt_ids", "max_tokens", "cancelled", "tokens", "done", "error",
                 "ignore_eos")

    def __init__(self, prompt_ids, max_tokens, cancelled, ignore_eos=False):
        self.prompt_ids, self.max_tokens, self.cancelled = prompt_ids, max_tokens, cancelled
        self.ignore_eos = bool(ignore_eos)
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

    greedy_only = True

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

    def encode(self, text): return self._encode(text)
    def decode(self, ids): return self._decode(ids)

    def _ensure_driver(self):
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
        pending_submits: list[tuple[list[int], int, bool]] = []
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
                                             ignore_eos=job.ignore_eos)
                except Exception as exc:  # e.g. prompt+max_tokens > max_seq_len
                    self._fail([job], exc)
                    continue
                self._active[rid] = job
                self._seen[rid] = 0
                pending_submits.append((job.prompt_ids, job.max_tokens, job.ignore_eos))
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

    def stream_token_ids(self, prompt_ids, max_tokens, cancelled, ignore_eos=False):
        self._ensure_driver()
        job = _Job(list(prompt_ids), int(max_tokens), cancelled,
                   ignore_eos=ignore_eos)
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
        self._stop.set()
        if self._sync_hook is not None:
            try:
                self._sync_hook([], False, shutdown=True)
            except Exception:
                pass
        driver = self._driver
        if driver is not None and driver.is_alive():
            driver.join(timeout=5.0)


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
            engine.submit(list(prompt_ids), int(max_new), ignore_eos=ignore_eos)
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
                                                  ignore_eos=ignore_eos)
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
                        self.send_header("Connection", "keep-alive")
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
                self.send_header("Connection", "keep-alive")
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

