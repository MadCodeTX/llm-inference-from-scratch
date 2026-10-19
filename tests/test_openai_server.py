"""OpenAI-compatible server contract tests (fake backend, CPU only).

Parses streaming SSE exactly like benchmarks/run_bench.py does, so the
shapes asserted here are the shapes the bench harness consumes.
"""
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from inference_from_scratch.openai_server import (BatchedEngineBackend,
                                                  FakeBackend, make_server)

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_bench", _REPO / "benchmarks" / "run_bench.py")
run_bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_bench)


@pytest.fixture()
def server():
    srv = make_server(FakeBackend(model_id="fake-model"), "127.0.0.1", 0,
                      request_timeout_s=30.0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def _post(url, payload, timeout=10):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _post_err(url, payload):
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(url, payload)
    return e.value.code, json.loads(e.value.read())


def _sse(url, payload):
    chunks, done = [], False
    with _post(url, payload) as resp:
        assert resp.headers["Content-Type"] == "text/event-stream"
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                done = True; break
            chunks.append(json.loads(body))
    return chunks, done


def test_health_and_models(server):
    with urllib.request.urlopen(server + "/health") as r:
        assert json.loads(r.read()) == {"status": "ok", "model": "fake-model"}
    with urllib.request.urlopen(server + "/v1/models") as r:
        body = json.loads(r.read())
    assert body["object"] == "list" and body["data"][0]["id"] == "fake-model"


def test_nonstreaming_shape_and_usage(server):
    with _post(server + "/v1/completions",
               {"model": "fake-model", "prompt": "hello", "max_tokens": 5,
                "temperature": 0.0}) as r:
        body = json.loads(r.read())
    ch = body["choices"][0]
    assert body["object"] == "text_completion" and body["model"] == "fake-model"
    assert ch["finish_reason"] == "length" and len(ch["text"]) == 5
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 5,
                             "total_tokens": 10}


def test_streaming_shape_run_bench_parser(server):
    chunks, done = _sse(server + "/v1/completions",
                        {"model": "fake-model", "prompt": "hi", "max_tokens": 4,
                         "temperature": 0.0, "stream": True,
                         "stream_options": {"include_usage": True}})
    assert done
    texts = [run_bench._chunk_text(c) for c in chunks]
    assert "".join(texts) == FakeBackend().decode(
        FakeBackend().encode(" alpha beta gamma delta"[:4])) or len("".join(texts)) == 4
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"] == {"prompt_tokens": 2, "completion_tokens": 4,
                                   "total_tokens": 6}
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])


def test_run_bench_one_request_against_fake(server):
    rec = run_bench.one_request(server, "fake-model", "hello world", 8,
                                {"temperature": 0.0, "top_p": 1.0})
    assert rec["ok"] and rec["error"] is None
    assert rec["completion_tokens"] == 8 and rec["prompt_tokens"] == 11
    assert rec["ttft_s"] is not None


def test_max_tokens_zero_and_stop_and_eos():
    srv = make_server(FakeBackend(model_id="fake-model", eos_id=0,
                                  scripted_ids=[ord("a"), ord("b"), 0, ord("c")]),
                      "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with _post(base + "/v1/completions",
                   {"model": "fake-model", "prompt": "x", "max_tokens": 10}) as r:
            body = json.loads(r.read())
        assert body["choices"][0]["text"] == "ab"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["completion_tokens"] == 2
        with _post(base + "/v1/completions",
                   {"model": "fake-model", "prompt": "x", "max_tokens": 0}) as r:
            body = json.loads(r.read())
        assert body["choices"][0]["text"] == "" and body["usage"]["completion_tokens"] == 0
    finally:
        srv.shutdown(); srv.server_close()


def test_stop_sequence(server):
    with _post(server + "/v1/completions",
               {"model": "fake-model", "prompt": "hi", "max_tokens": 32,
                "stop": ["beta"]}) as r:
        body = json.loads(r.read())
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "beta" not in body["choices"][0]["text"]
    assert body["choices"][0]["text"] == " alpha "


def test_fake_sampled_seeded_reproducible(server):
    payload = {"model": "fake-model", "prompt": "hi", "max_tokens": 8,
               "temperature": 0.7, "top_p": 0.8, "seed": 1234}
    with _post(server + "/v1/completions", payload) as r:
        a = json.loads(r.read())
    with _post(server + "/v1/completions", payload) as r:
        b = json.loads(r.read())
    assert a["choices"][0]["text"] == b["choices"][0]["text"]
    assert a["usage"]["completion_tokens"] == 8
    rec = run_bench.one_request(server, "fake-model", "hello world", 8,
                                {"temperature": 0.7, "top_p": 0.8, "seed": 1234})
    assert rec["ok"] and rec["completion_tokens"] == 8


def test_sampled_validation_and_greedy_only_backend(server):
    for field, payload in [
            ("top_p", {"temperature": 0.7, "top_p": 0.0}),
            ("top_p", {"temperature": 0.7, "top_p": 1.5}),
            ("top_k", {"temperature": 0.7, "top_k": 0}),
            ("seed", {"temperature": 0.7, "seed": -1}),
            ("seed", {"temperature": 0.7, "seed": "x"})]:
        code, body = _post_err(server + "/v1/completions",
                               {"model": "fake-model", "prompt": "x",
                                "max_tokens": 2, **payload})
        assert code == 400 and body["error"]["param"] == field, payload

    class _GreedyOnly(FakeBackend):
        greedy_only = True

    srv = make_server(_GreedyOnly(model_id="fake-model"), "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    try:
        code, body = _post_err(
            f"http://127.0.0.1:{srv.server_address[1]}/v1/completions",
            {"model": "fake-model", "prompt": "x", "temperature": 0.7})
        assert code == 400 and body["error"]["param"] == "temperature"
    finally:
        srv.shutdown(); srv.server_close()


def test_errors(server):
    code, body = _post_err(server + "/v1/completions",
                           {"model": "nope", "prompt": "x", "max_tokens": 1})
    assert code == 404 and body["error"]["code"] == "model_not_found"
    code, _ = _post_err(server + "/v1/completions",
                        {"model": "fake-model", "prompt": "", "max_tokens": 1})
    assert code == 400
    code, _ = _post_err(server + "/v1/completions",
                        {"model": "fake-model", "prompt": "x", "max_tokens": -1})
    assert code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(server + "/nope")
    assert e.value.code == 404


def test_concurrent_isolation(server):
    def one(i):
        with _post(server + "/v1/completions",
                   {"model": "fake-model", "prompt": "p" * (i + 1),
                    "max_tokens": 6, "temperature": 0.0}) as r:
            return json.loads(r.read())
    with ThreadPoolExecutor(8) as ex:
        bodies = list(ex.map(one, range(8)))
    assert all(b["usage"]["completion_tokens"] == 6 for b in bodies)
    assert [b["usage"]["prompt_tokens"] for b in bodies] == list(range(1, 9))
    assert len({b["id"] for b in bodies}) == 8


class _StubEngine:
    """TPEngineCore-shaped stub: token 2 of every request is EOS (99)."""

    eos_id = 99

    def __init__(self, fail_submit=False, fail_step=False):
        self._reqs, self._next = {}, 0
        self.fail_submit, self.fail_step = fail_submit, fail_step
        self.sampling_seen = []

    def submit(self, prompt_ids, max_new, ignore_eos=False, temperature=0.0,
               top_p=1.0, top_k=None, seed=None):
        if self.fail_submit:
            raise ValueError("prompt + generated needs 999 > max_seq_len=8")
        self.sampling_seen.append({"temperature": temperature, "top_p": top_p,
                                   "top_k": top_k, "seed": seed})
        rid, self._next = self._next, self._next + 1
        self._reqs[rid] = {"ids": [], "max": max_new, "ignore": ignore_eos,
                           "done": max_new == 0}
        return rid

    def step(self):
        if self.fail_step:
            raise RuntimeError("engine step exploded")
        for r in self._reqs.values():
            if r["done"]:
                continue
            tok = 99 if len(r["ids"]) == 1 else 65 + len(r["ids"])
            r["ids"].append(tok)
            if len(r["ids"]) >= r["max"] or (tok == 99 and not r["ignore"]):
                r["done"] = True

    def poll(self, rid):
        r = self._reqs[rid]
        return {"status": "finished" if r["done"] else "decoding",
                "token_ids": list(r["ids"])}


def _batched(eng):
    return BatchedEngineBackend(
        "fake-model", eng, lambda t: [ord(c) for c in t],
        lambda ids: "".join(chr(i) for i in ids if i != 99), eos_id=99)


def test_batched_backend_eos_and_ignore_eos():
    be = _batched(_StubEngine())
    try:
        got = list(be.stream_token_ids([120], 4, threading.Event()))
        assert got == [65, 99]  # engine EOS finishes the request
        got = list(be.stream_token_ids([120], 4, threading.Event(),
                                       ignore_eos=True))
        assert got == [65, 99, 67, 68]  # ignore_eos reaches the engine
    finally:
        be.close()


def test_batched_backend_driver_error_does_not_hang():
    for eng in (_StubEngine(fail_submit=True), _StubEngine(fail_step=True)):
        be = _batched(eng)
        try:
            with pytest.raises((ValueError, RuntimeError)):
                list(be.stream_token_ids([120], 2, threading.Event()))
        finally:
            be.close()
            assert be._driver is None or not be._driver.is_alive()


def test_batched_backend_no_per_token_poll_latency():
    # Instant stub engine: with the old always-blocking 50 ms poll,
    # 32 tokens took ~1.6 s. Active work must step immediately.
    import time as _t
    be = _batched(_StubEngine())
    try:
        t0 = _t.perf_counter()
        got = list(be.stream_token_ids([120], 32, threading.Event(),
                                       ignore_eos=True))
        elapsed = _t.perf_counter() - t0
        assert len(got) == 32 and elapsed < 0.5, elapsed
    finally:
        be.close()


def test_batched_backend_idle_driver_blocks_and_close_is_prompt():
    # Idle driver must not busy-step the engine, and close() must wake
    # it promptly (within a few idle-poll quanta, not a hang).
    import time as _t
    eng = _StubEngine()
    steps = []
    orig_step = eng.step
    eng.step = lambda: (steps.append(1), orig_step())[1]
    be = _batched(eng)
    be._ensure_driver()
    _t.sleep(0.25)
    assert steps == []  # no busy-spin stepping while idle
    t0 = _t.perf_counter()
    be.close()
    assert _t.perf_counter() - t0 < 2.0
    assert be._driver is not None and not be._driver.is_alive()


def test_batched_backend_cancel_is_prompt():
    import time as _t

    class Slow(_StubEngine):
        def step(self):
            _t.sleep(0.02)
            super().step()

    be = _batched(Slow())
    cancelled = threading.Event()
    out = {}

    def consume():
        out["got"] = list(be.stream_token_ids([120], 100, cancelled,
                                              ignore_eos=True))

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    _t.sleep(0.1)
    cancelled.set()
    t.join(timeout=3)
    try:
        assert not t.is_alive()  # consumer returns instead of hanging
        assert len(out.get("got", [])) < 100
    finally:
        be.close()


def test_batched_backend_sixteen_concurrent_contract():
    # c16-shaped stub contract: 16 concurrent jobs through one driver
    # all finish with their exact tokens; none error (HTTP 500 class).
    be = _batched(_StubEngine())
    try:
        with ThreadPoolExecutor(16) as ex:
            results = list(ex.map(
                lambda _i: list(be.stream_token_ids(
                    [120], 4, threading.Event(), ignore_eos=True)),
                range(16)))
        assert results == [[65, 99, 67, 68]] * 16
    finally:
        be.close()


def test_batched_backend_close_wakes_idle_driver():
    be = _batched(_StubEngine())
    be._ensure_driver()
    be.close()
    assert be._driver is not None and not be._driver.is_alive()


def test_batched_backend_http_ignore_eos():
    be = _batched(_StubEngine())
    srv = make_server(be, "127.0.0.1", 0, request_timeout_s=30.0)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with _post(base + "/v1/completions",
                   {"model": "fake-model", "prompt": "x", "max_tokens": 4,
                    "ignore_eos": True}) as r:
            body = json.loads(r.read())
        assert body["choices"][0]["finish_reason"] == "length"
        assert body["usage"]["completion_tokens"] == 4
        with _post(base + "/v1/completions",
                   {"model": "fake-model", "prompt": "x", "max_tokens": 4}) as r:
            body = json.loads(r.read())
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["completion_tokens"] == 1
    finally:
        srv.shutdown(); srv.server_close(); be.close()


def test_batched_backend_sampled_sampling_reaches_engine():
    eng = _StubEngine()
    be = _batched(eng)
    srv = make_server(be, "127.0.0.1", 0, request_timeout_s=30.0)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with _post(base + "/v1/completions",
                   {"model": "fake-model", "prompt": "x", "max_tokens": 2,
                    "ignore_eos": True,
                    "temperature": 0.7, "top_p": 0.8, "top_k": 5,
                    "seed": 1234}) as r:
            body = json.loads(r.read())
        assert body["usage"]["completion_tokens"] == 2
    finally:
        srv.shutdown(); srv.server_close(); be.close()
    assert eng.sampling_seen == [{"temperature": 0.7, "top_p": 0.8,
                                  "top_k": 5, "seed": 1234}]


def test_lockstep_codec_sampled_roundtrip():
    from inference_from_scratch.openai_server import (
        decode_lockstep_command, encode_lockstep_command)
    sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 5, "seed": 1234}
    enc = encode_lockstep_command([([7, 8], 4, False, sampling)], True)
    subs, do_step, shutdown = decode_lockstep_command(enc)
    assert (do_step, shutdown) == (True, False)
    assert subs == [([7, 8], 4, False, sampling)]
    # Greedy encoding is unchanged (no trailing block).
    assert encode_lockstep_command([([7, 8], 4, False)], True) == [
        0, 1, 1, 2, 4, 0, 7, 8]
    with pytest.raises(ValueError):
        decode_lockstep_command(enc[:-1])


def test_lockstep_codec_sampling_floats_exact():
    from inference_from_scratch.openai_server import (
        decode_lockstep_command, encode_lockstep_command)
    for sampling in ({"temperature": 0.3333333, "top_p": 0.1234567,
                      "top_k": None, "seed": None},
                     {"temperature": 1e-7, "top_p": 1e-7,
                      "top_k": 1, "seed": 0}):
        subs, _, _ = decode_lockstep_command(
            encode_lockstep_command([([7], 2, False, sampling)], True))
        assert subs[0][3] == sampling


def test_batched_backend_close_fails_inflight_and_idempotent():
    import time as _t
    class Slow(_StubEngine):
        def step(self):
            _t.sleep(0.3); super().step()
    calls = []
    be = BatchedEngineBackend(
        "fake-model", Slow(), lambda t: [ord(c) for c in t],
        lambda ids: "", eos_id=99,
        sync_hook=lambda s, d, shutdown=False: calls.append(shutdown))
    out = {}
    def consume():
        try:
            out["got"] = list(be.stream_token_ids([120], 4, threading.Event()))
        except Exception as e:  # noqa: BLE001
            out["err"] = e
    t = threading.Thread(target=consume, daemon=True); t.start()
    _t.sleep(0.1)  # driver is mid-step with the job in-flight
    be.close(); be.close()
    t.join(timeout=5)
    assert not t.is_alive()
    assert isinstance(out.get("err"), RuntimeError)
    assert calls.count(True) == 1 and calls[-1] is True  # one shutdown


def test_lockstep_codec_roundtrip_and_truncation():
    from inference_from_scratch.openai_server import (
        decode_lockstep_command, encode_lockstep_command)
    assert encode_lockstep_command([], True) == [0, 1, 0]
    assert encode_lockstep_command([], False, shutdown=True) == [1, 0, 0]
    subs = [([7, 8, 9], 4, True), ([1], 0, False)]
    enc = encode_lockstep_command(subs, True)
    assert decode_lockstep_command(enc) == (subs, True, False)
    assert decode_lockstep_command(encode_lockstep_command([], False, True)) == ([], False, True)
    with pytest.raises(ValueError):
        decode_lockstep_command(enc[:-1])
    with pytest.raises(ValueError):
        decode_lockstep_command([0, 1])


def test_participate_compact_mirrors_and_shutdown_and_errors():
    import queue as _q
    from inference_from_scratch.openai_server import (
        encode_lockstep_command, participate_compact)
    eng = _StubEngine()
    cmds = _q.Queue()
    for c in [encode_lockstep_command([([120], 3, True)], True),
              encode_lockstep_command([], True),
              encode_lockstep_command([], True),
              encode_lockstep_command([], False, shutdown=True)]:
        cmds.put(c)
    participate_compact(eng, cmds.get)  # returns only on shutdown command
    assert eng.poll(0)["token_ids"] == [65, 99, 67]
    bad = _StubEngine(fail_step=True)
    cmds2 = _q.Queue()
    cmds2.put(encode_lockstep_command([([120], 2, False)], True))
    with pytest.raises(RuntimeError):
        participate_compact(bad, cmds2.get)


def test_participate_compact_sampled_mirror():
    import queue as _q
    from inference_from_scratch.openai_server import (
        encode_lockstep_command, participate_compact)
    eng = _StubEngine()
    cmds = _q.Queue()
    sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": None, "seed": 1234}
    cmds.put(encode_lockstep_command([([120], 2, False, sampling)], True))
    cmds.put(encode_lockstep_command([], False, shutdown=True))
    participate_compact(eng, cmds.get)
    assert eng.sampling_seen == [sampling]


def test_batched_backend_compact_sync_counts_and_mirror():
    # Leader backend drives a worker stub via compact commands; asserts
    # one sync per engine step plus one shutdown, exact mirrored tokens.
    import queue as _q
    from inference_from_scratch.openai_server import (
        encode_lockstep_command, participate_compact)
    worker = _StubEngine()
    cmds: _q.Queue = _q.Queue()
    calls = []
    def hook(submits, do_step, shutdown=False):
        enc = encode_lockstep_command(submits, do_step, shutdown)
        calls.append((len(submits), do_step, shutdown, len(enc)))
        cmds.put(enc)
    wt = threading.Thread(target=participate_compact, args=(worker, cmds.get), daemon=True)
    wt.start()
    be = BatchedEngineBackend("fake-model", _StubEngine(),
                              lambda t: [ord(c) for c in t],
                              lambda ids: "", eos_id=99, sync_hook=hook)
    try:
        got = list(be.stream_token_ids([120], 4, threading.Event(), ignore_eos=True))
        assert got == [65, 99, 67, 68]
    finally:
        be.close()
    wt.join(timeout=5)
    assert not wt.is_alive()  # shutdown broadcast freed the worker
    assert worker.poll(0)["token_ids"] == got
    assert calls[-1] == (0, False, True, 3)  # compact shutdown: 3 ints
    assert all(c[3] == 3 for c in calls if c[0] == 0 and c[1])  # step-only: 3 ints
    assert sum(1 for c in calls if c[1]) == 4  # one sync per engine step


def test_streaming_raw_socket_eof_after_done(server):
    import socket
    host, port = server.removeprefix("http://").split(":")
    body = json.dumps({"model": "fake-model", "prompt": "hi",
                       "max_tokens": 2, "stream": True}).encode()
    with socket.create_connection((host, int(port)), timeout=5) as s:
        s.settimeout(5)
        s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\n"
                  b"Content-Type: application/json\r\nContent-Length: "
                  + str(len(body)).encode() + b"\r\n\r\n" + body)
        data = b""
        while b"[DONE]" not in data:
            chunk = s.recv(4096)
            assert chunk, "connection closed before [DONE]"
            data += chunk
        # After [DONE] the server must close: next recv is EOF, not a hang.
        assert s.recv(4096) == b""
    assert b"connection: close" in data.lower()


def test_quality_probe_nonstreaming_fields(server):
    # quality_probe.py sends these extra fields; they must not break us.
    with _post(server + "/v1/completions",
               {"model": "fake-model", "prompt": "The capital of France is",
                "max_tokens": 4, "temperature": 0.0, "repetition_penalty": 1.0,
                "ignore_eos": False, "logprobs": 1, "seed": 42}) as r:
        body = json.loads(r.read())
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["prompt_tokens"] == len("The capital of France is")
