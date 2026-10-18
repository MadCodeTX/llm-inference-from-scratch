"""GPU smoke test for the tiny hybrid stack (CUDA only, tiny dims).

Builds the tiny hybrid config used in ``tests/test_hybrid.py`` (3 linear
+ 1 full layer, D=32, vocab 97) with a fixed seed and runs, on
``cuda:0``:

(a) dense prefill + ``decode_step`` loop in fp32, compared token- and
    logit-wise against the same-seed CPU model;
(b) the same prefill in bf16, reporting the honest diff against the
    fp32 CPU reference (bf16 is *expected* to differ; token agreement
    is counted, not assumed);
(c) a ``StaticHybridState`` ``decode_token_static`` loop on CUDA
    against the legacy step path on the same device;
(d) chunkwise prefill on CUDA against sequential prefill on CUDA.

Also (e) instantiates the REAL Qwen3.8 ``HybridLM`` config on the
``meta`` device (no allocation) and reports its parameter count.

Every number printed comes from the run itself; settings (torch /
CUDA versions, device name, seed, shapes, dtypes) are printed first.
fp32 checks (a/c/d) fail the script (exit 1) past 1e-5 max-abs diff;
bf16 (b) is report-only beyond finiteness. Peak memory is
``torch.cuda.max_memory_allocated`` over the CUDA sections.

Usage: .venv-gpu/bin/python scripts/gpu_smoke_hybrid.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_from_scratch.hybrid import (  # noqa: E402
    HybridConfig, HybridLM, TinyHybridLM, generate_hybrid,
)

D, VOCAB = 32, 97
SEED = 3
TOL_FP32 = 1e-5


def _tiny_config(**overrides):
    kwargs = dict(
        vocab_size=VOCAB, hidden_size=D, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128,
        linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8,
        partial_rotary_factor=0.25, rope_theta=10000.0,
    )
    kwargs.update(overrides)
    return HybridConfig(**kwargs)


def _tiny_lm(seed=SEED):
    torch.manual_seed(seed)
    m = TinyHybridLM(_tiny_config())
    m.eval()
    return m


def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA unavailable; nothing to smoke-test")
        return 1
    dev = torch.device(args.device)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={dev} ({torch.cuda.get_device_name(dev)}) seed={args.seed}")
    print(f"tiny config: D={D} VOCAB={VOCAB} layers=3xlinear+1xfull "
          f"prompt=[2, 7] fp32 tol={TOL_FP32}")
    failures: list[str] = []

    def check(name: str, ok: bool) -> None:
        print(f"  -> {'PASS' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append(name)

    # Same-seed CPU reference and CUDA model (init runs on CPU in both
    # builds, so weights are identical; verified via state_dict below).
    cpu = _tiny_lm(args.seed)
    gpu = _tiny_lm(args.seed).to(dev)
    same_w = all(torch.equal(a, b.cpu())
                 for a, b in zip(cpu.state_dict().values(),
                                 gpu.state_dict().values()))
    print(f"weights identical cpu/gpu: {same_w}")
    torch.manual_seed(1234)
    idx_cpu = torch.randint(0, VOCAB, (2, 7))
    idx = idx_cpu.to(dev)
    with torch.no_grad():
        full_cpu, _ = cpu(idx_cpu)

    torch.cuda.reset_peak_memory_stats(dev)
    # (a) dense prefill + decode on CUDA fp32 vs CPU.
    with torch.no_grad():
        full_gpu, _ = gpu(idx)
        _, state = gpu.prefill(idx[:, :4])
        steps = []
        for t in range(4, 7):
            logits, state = gpu.decode_step(idx[:, t], state)
            steps.append(logits)
        joined = torch.cat(steps, dim=1)
    mx, mean = _diff(full_gpu.cpu(), full_cpu)
    tok_eq = torch.equal(full_gpu.argmax(-1).cpu(), full_cpu.argmax(-1))
    print(f"(a) prefill fp32 cuda vs cpu: max_abs={mx:.3e} mean_abs={mean:.3e} "
          f"tokens_equal={tok_eq}")
    check("(a) prefill logits", mx < TOL_FP32 and tok_eq)
    mx, mean = _diff(joined.cpu(), full_cpu[:, 4:])
    tok_eq = torch.equal(joined.argmax(-1).cpu(), full_cpu[:, 4:].argmax(-1))
    print(f"(a) decode steps vs cpu prefill tail: max_abs={mx:.3e} "
          f"mean_abs={mean:.3e} tokens_equal={tok_eq}")
    check("(a) decode logits", mx < TOL_FP32 and tok_eq)
    # Sampled decode with a plain CPU generator must work on CUDA too.
    with torch.no_grad():
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        s1 = generate_hybrid(gpu, idx[:, :4].clone(), 3, temperature=0.8,
                             top_k=10, generator=g1)
        s2 = generate_hybrid(gpu, idx[:, :4].clone(), 3, temperature=0.8,
                             top_k=10, generator=g2)
    print(f"(a) sampled generate w/ CPU generator: shape={tuple(s1.shape)} "
          f"reproducible={torch.equal(s1, s2)} device={s1.device}")
    check("(a) CPU-generator sampling", torch.equal(s1, s2))

    # (b) bf16: report honestly, including token disagreement.
    gpu16 = _tiny_lm(args.seed).to(device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        full16, _ = gpu16(idx)
    mx, mean = _diff(full16.cpu(), full_cpu)
    agree = (full16.argmax(-1).cpu() == full_cpu.argmax(-1))
    print(f"(b) bf16 cuda vs fp32 cpu: max_abs={mx:.3e} mean_abs={mean:.3e} "
          f"token_agreement={int(agree.sum())}/{agree.numel()} "
          f"finite={bool(torch.isfinite(full16).all())}")
    check("(b) bf16 finite", bool(torch.isfinite(full16).all()))

    # (c) StaticHybridState decode loop vs legacy step path, both CUDA.
    with torch.no_grad():
        legacy_state = gpu.initial_state(2)
        legacy = []
        for t in range(7):
            logits, legacy_state = gpu.step(idx[:, t], legacy_state)
            legacy.append(logits[:, 0, :])
        legacy = torch.stack(legacy, dim=1)
        st = gpu.static_state(batch_size=2, max_seq_len=16)
        static = torch.stack(
            [gpu.decode_token_static(idx[:, t], st) for t in range(7)], dim=1)
    mx, mean = _diff(static, legacy)
    print(f"(c) static vs legacy (cuda): max_abs={mx:.3e} mean_abs={mean:.3e} "
          f"state_device={st.kv_cache.k_cache.device} seq_len={st.seq_len}")
    check("(c) static logits", mx < TOL_FP32 and st.seq_len == 7)

    # (d) chunkwise prefill vs sequential prefill, both CUDA.
    with torch.no_grad():
        seq_logits, _ = gpu(idx)
        chu_logits, _ = gpu.prefill(idx, prefill_method="chunkwise",
                                    chunk_size=4)
    mx, mean = _diff(chu_logits, seq_logits)
    print(f"(d) chunkwise vs sequential (cuda): max_abs={mx:.3e} "
          f"mean_abs={mean:.3e}")
    check("(d) chunkwise logits", mx < TOL_FP32)

    torch.cuda.synchronize(dev)
    peak = torch.cuda.max_memory_allocated(dev)
    print(f"peak_memory_allocated: {peak} B ({peak / 2**20:.2f} MiB) on {dev}")

    # (e) REAL config on meta: parameter count only, no allocation.
    snap = _find_snapshot()
    if snap is None:
        print("(e) real snapshot not found in HF cache; skipped")
    else:
        cfg = HybridConfig.from_text_config(snap)
        with torch.device("meta"):
            real = HybridLM(cfg)
        n = sum(p.numel() for p in real.parameters())
        print(f"(e) real HybridLM on meta: snapshot={snap.name} "
              f"layers={cfg.num_hidden_layers} "
              f"({cfg.num_linear_layers} linear + {cfg.num_full_layers} full) "
              f"parameter_count={n}")
    print("RESULT:", "FAIL " + ", ".join(failures) if failures else "PASS")
    return 1 if failures else 0


def _find_snapshot() -> Path | None:
    hub = (Path.home() / ".cache" / "huggingface" / "hub"
           / "models--Qwen--Qwen3.8-27B-FP8" / "snapshots")
    if hub.is_dir():
        for snap in sorted(hub.iterdir()):
            if (snap / "config.json").is_file():
                return snap
    return None


if __name__ == "__main__":
    raise SystemExit(main())
