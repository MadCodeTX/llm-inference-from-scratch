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


def test_errors(server):
    code, body = _post_err(server + "/v1/completions",
                           {"model": "nope", "prompt": "x", "max_tokens": 1})
    assert code == 404 and body["error"]["code"] == "model_not_found"
    code, body = _post_err(server + "/v1/completions",
                           {"model": "fake-model", "prompt": "x",
                            "temperature": 0.7})
    assert code == 400 and body["error"]["param"] == "temperature"
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

    def submit(self, prompt_ids, max_new, ignore_eos=False):
        if self.fail_submit:
            raise ValueError("prompt + generated needs 999 > max_seq_len=8")
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


def test_quality_probe_nonstreaming_fields(server):
    # quality_probe.py sends these extra fields; they must not break us.
    with _post(server + "/v1/completions",
               {"model": "fake-model", "prompt": "The capital of France is",
                "max_tokens": 4, "temperature": 0.0, "repetition_penalty": 1.0,
                "ignore_eos": False, "logprobs": 1, "seed": 42}) as r:
        body = json.loads(r.read())
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["prompt_tokens"] == len("The capital of France is")
