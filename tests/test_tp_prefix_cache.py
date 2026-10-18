"""Block prefix cache for TPStaticState: hasher, cache, integration, bytes."""
import subprocess, sys
from pathlib import Path
import pytest, torch
ROOT = Path(__file__).resolve().parents[1]
from inference_from_scratch.hybrid import HybridConfig, TinyHybridLM
from inference_from_scratch.tensor_parallel import state_bytes as tp_state_bytes
from inference_from_scratch.tp_distributed import DistributedHybridLM, _LocalContext
from inference_from_scratch.tp_prefix_cache import (BlockHasher, TPBlockPrefixCache,
    prefill_with_prefix_cache, state_bytes as cache_state_bytes, ROOT_HASH, entry_key)

def _tiny_cfg():
    return HybridConfig(vocab_size=97, hidden_size=32, num_hidden_layers=4,
        layer_types=["linear_attention","linear_attention","linear_attention","full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=64,
        max_position_embeddings=128, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_num_value_heads=4, linear_value_head_dim=8, partial_rotary_factor=0.25, rope_theta=10000.0)

def _tp():
    torch.manual_seed(7); dense = TinyHybridLM(_tiny_cfg()); dense.eval()
    tp = DistributedHybridLM.from_dense(dense, _LocalContext(0,1,torch.device("cpu")), config=_tiny_cfg()); tp.eval(); return tp

def test_hasher_deterministic_chain():
    h = BlockHasher(16)
    a = list(range(32)); b = list(range(16)) + [999]*16
    ha, hb = h.chain_hashes(a), h.chain_hashes(b)
    assert ha[0] == hb[0] and ha[1] != hb[1]  # chain: first block same, second differs
    assert BlockHasher(16).chain_hashes(a) == ha  # deterministic across instances (not hash())
    assert h.hash_block(ROOT_HASH, a[:16]) == ha[0]
    with pytest.raises(ValueError): h.hash_block(ROOT_HASH, [1,2])

def test_two_prompts_shared_prefix_exact_and_reuse():
    tp = _tp(); torch.manual_seed(99)
    shared = torch.randint(0,97,(1,32)); promptA = torch.cat([shared, torch.randint(0,97,(1,13))],1)
    promptB = torch.cat([shared, torch.randint(0,97,(1,13))],1)  # partial-block suffix (13)
    with torch.no_grad():
        cache = TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=16)
        sA = tp.static_state(batch_size=1, max_seq_len=64); lA, nA, _ = prefill_with_prefix_cache(tp, promptA, sA, cache, chunk_size=4)
        assert nA == 0
        cacheF = TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=16)
        sF = tp.static_state(batch_size=1, max_seq_len=64); lF, _, _ = prefill_with_prefix_cache(tp, promptB, sF, cacheF, chunk_size=4)
        sB = tp.static_state(batch_size=1, max_seq_len=64); lB, nB, dB = prefill_with_prefix_cache(tp, promptB, sB, cache, chunk_size=4)
        assert nB == 32 and dB["tokens_saved"] == 32
        assert torch.equal(lB, lF)  # EXACTLY equal fresh
        nxtB, nxtF = lB.argmax(-1), lF.argmax(-1)
        for _ in range(4):
            oB, oF = tp.decode_token_static(nxtB, sB), tp.decode_token_static(nxtF, sF)
            assert torch.equal(oB, oF); nxtB, nxtF = oB.argmax(-1), oF.argmax(-1)
        assert cache.stats()["hits"] >= 2

def test_aliasing_and_eviction_and_stats():
    tp = _tp(); torch.manual_seed(5)
    with torch.no_grad():
        cache = TPBlockPrefixCache(block_size=16, max_entries=2, snapshot_interval_tokens=16)
        s = tp.static_state(batch_size=1, max_seq_len=64)
        ids = torch.randint(0,97,(1,48))
        _, _, _ = prefill_with_prefix_cache(tp, ids, s, cache, chunk_size=4)
        assert len(cache) == 2  # LRU eviction by entries: first block evicted
        hashes = BlockHasher(16).chain_hashes(ids[0].tolist())
        assert cache.get(hashes[0]) is None and cache.get(hashes[-1]) is not None
        # Aliasing: mutate live state, cached snapshot unchanged; restore recovers.
        e = cache.get(hashes[-1]); snap_S = e.linear_S[0].clone(); snap_pos = e.position
        s.linear_S[0].fill_(123.0); s.kv_k[3].fill_(9.0)
        assert torch.equal(cache.get(hashes[-1]).linear_S[0], snap_S)
        cache.restore(hashes[-1], s)
        assert s.seq_len == snap_pos == 48 and torch.equal(s.linear_S[0], snap_S)
        # Byte cap eviction
        cap = cache_state_bytes(s, filled=48)
        cache2 = TPBlockPrefixCache(block_size=16, max_bytes=cap, snapshot_interval_tokens=16)
        s2 = tp.static_state(batch_size=1, max_seq_len=64)
        prefill_with_prefix_cache(tp, ids, s2, cache2, chunk_size=4)
        assert len(cache2) == 1 and cache2.total_bytes <= cap

def test_byte_accounting_vs_formula_real_config():
    cfg = HybridConfig()  # real Qwen3.8 config
    world = 2
    f16 = tp_state_bytes(cfg, seq_len=16, batch=1, kv_dtype_bytes=2, world=world)
    f32 = tp_state_bytes(cfg, seq_len=32, batch=1, kv_dtype_bytes=2, world=world)
    marginal = f32["full_kv_bytes_per_rank"] - f16["full_kv_bytes_per_rank"]
    linear = f16["linear_bytes_per_rank"]
    # Our measured snapshot growth 16->32 on a real-shaped state would be marginal;
    # formula check: marginal = n_full*2*B*16*(Hkv/world)*Dh*2
    # tensor_parallel._dims maps default HybridConfig head_dim to real 256
    assert marginal == f16["full_layers"]*2*1*16*f16["kv_heads_per_rank"]*256*2
    print(f"per-16-token-block marginal bytes/rank (world={world}): {marginal}")
    print(f"per-snapshot linear-state bytes/rank: {linear}")
    assert marginal > 0 and linear > 0
    # Tiny measured: snapshot at 32 minus at 16 equals KV-only growth
    tp = _tp()
    with torch.no_grad():
        s = tp.static_state(batch_size=1, max_seq_len=64)
        tp.prefill_chunkwise_static(torch.randint(0,97,(1,32)), s, chunk_size=4)
        assert cache_state_bytes(s, filled=32) - cache_state_bytes(s, filled=16) > 0
        assert cache_state_bytes(s, filled=32) - cache_state_bytes(s, filled=16) == sum(
            s.kv_k[i][:, :, :16].numel()*s.kv_k[i].element_size()*2 for i in range(4) if s.kv_k[i] is not None)

def test_snapshot_interval_validation_and_keys():
    with pytest.raises(ValueError): TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=24)
    with pytest.raises(ValueError): TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=0)
    c = TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=256)
    assert c.snapshot_interval_tokens == 256 and "@" in entry_key(b"\x00"*32, 256)

def test_snapshot_interval_2304_count_and_shared2048_reuse():
    cfg = _tiny_cfg(); cfg = type(cfg)(**{**cfg.__dict__, "max_position_embeddings": 8192}) if hasattr(cfg, "__dict__") else cfg
    torch.manual_seed(7); dense = TinyHybridLM(cfg); dense.eval()
    tp = DistributedHybridLM.from_dense(dense, _LocalContext(0,1,torch.device("cpu")), config=cfg); tp.eval()
    torch.manual_seed(123)
    shared = torch.randint(0,97,(1,2048)); promptA = torch.cat([shared, torch.randint(0,97,(1,256))],1)
    promptB = torch.cat([shared, torch.randint(0,97,(1,256))],1)
    with torch.no_grad():
        cache = TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=256)
        sA = tp.static_state(batch_size=1, max_seq_len=2560); _, nA, dA = prefill_with_prefix_cache(tp, promptA, sA, cache, chunk_size=128)
        assert nA == 0
        # 9 interval boundaries (256..2304) + 1 final event (deduplicated
        # at the same boundary) = 10 snapshot events, 9 distinct entries.
        assert dA["blocks_stored"] == 10 and dA["snapshots"] == 10, dA
        assert len(cache) == 9 and cache.stored_lengths() == [256*i for i in range(1,10)]
        assert cache.stats()["snapshot_bytes_copied"] == cache.total_bytes
        sB = tp.static_state(batch_size=1, max_seq_len=2560); lB, nB, _ = prefill_with_prefix_cache(tp, promptB, sB, cache, chunk_size=128)
        assert nB == 2048  # shared 2048 reused exactly (final of A at 2304 not used)
        cacheF = TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=256)
        sF = tp.static_state(batch_size=1, max_seq_len=2560); lF, _, _ = prefill_with_prefix_cache(tp, promptB, sF, cacheF, chunk_size=128)
        assert torch.equal(lB, lF)

def test_interval_partial_reuse_exact():
    tp = _tp(); torch.manual_seed(77)
    shared300 = torch.randint(0,97,(1,300))
    promptA = torch.cat([shared300, torch.randint(0,97,(1,212))],1)  # 512
    promptB = torch.cat([shared300, torch.randint(0,97,(1,100))],1)  # shares 300, len 400
    with torch.no_grad():
        cache = TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=256)
        sA = tp.static_state(batch_size=1, max_seq_len=640); prefill_with_prefix_cache(tp, promptA, sA, cache, chunk_size=16)
        assert cache.stored_lengths() == [256, 512]
        sB = tp.static_state(batch_size=1, max_seq_len=640); lB, nB, _ = prefill_with_prefix_cache(tp, promptB, sB, cache, chunk_size=16)
        assert nB == 256  # longest stored <= shared 300; 44 shared tokens recomputed
        cacheF = TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=256)
        sF = tp.static_state(batch_size=1, max_seq_len=640); lF, _, _ = prefill_with_prefix_cache(tp, promptB, sF, cacheF, chunk_size=16)
        assert torch.equal(lB, lF)
        # store_final=False: a 300-token prompt stores only the 256 boundary
        c2 = TPBlockPrefixCache(block_size=16, snapshot_interval_tokens=256, store_final=False)
        s2 = tp.static_state(batch_size=1, max_seq_len=640); prefill_with_prefix_cache(tp, promptA[:, :300], s2, c2, chunk_size=16)
        assert c2.stored_lengths() == [256]

@pytest.mark.skipif(not (ROOT/"scripts/tp2_prefix_cache_check.py").exists(), reason="missing")
def test_tp2_prefix_cache_gloo():
    proc = subprocess.run([sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=2",
        str(ROOT/"scripts/tp2_prefix_cache_check.py"),"--backend","gloo","--device","cpu"],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert proc.returncode==0, proc.stdout[-3000:]+proc.stderr[-3000:]
    assert "TP2_PREFIX_CACHE_CHECK PASS" in proc.stdout and "n_reused=32" in proc.stdout
