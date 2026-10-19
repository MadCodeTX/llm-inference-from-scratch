"""CPU tests for fixed FP8 tile config selection (no autotune)."""
import pytest
import inference_from_scratch.fp8_triton as ft
from inference_from_scratch.fp8_triton import (
    TUNED_CONFIGS, choose_config, m_bucket, set_config_mode, validate_config)


def test_m_bucket():
    assert m_bucket(1) == 1 and m_bucket(2) == 8 and m_bucket(8) == 8
    assert m_bucket(9) == 16 and m_bucket(32) == 32 and m_bucket(33) == 64
    assert m_bucket(64) == 64 and m_bucket(65) == 128 and m_bucket(128) == 128
    assert m_bucket(129) == 512 and m_bucket(512) == 512
    assert m_bucket(513) == 1024 and m_bucket(4096) == 1024


def test_tuned_exact_shapes():
    for (bucket, n, k), cfg in TUNED_CONFIGS.items():
        assert bucket == 1
        got = choose_config(1, n, k, mode="new")
        assert got == cfg and got["BLOCK_N"] == 32 and got["num_warps"] == 4


def test_m1_family_and_fallback():
    # Unlisted M=1 shape uses measured M=1 family config.
    assert choose_config(1, 4096, 4096, mode="new")["BLOCK_N"] == 32
    # Other buckets fall back to legacy defaults.
    assert choose_config(5, 5120, 17408, mode="new") == choose_config(5, 5120, 17408, mode="old")
    assert choose_config(128, 5120, 5120, mode="new")["BLOCK_M"] == 64


def test_old_mode_is_legacy_everywhere():
    assert choose_config(1, 17408, 5120, mode="old") == {
        "BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3}


def test_mode_switch_and_validation():
    try:
        assert set_config_mode("old") == "old"
        assert choose_config(1, 17408, 5120)["BLOCK_N"] == 128
        assert set_config_mode("new") == "new"
        assert choose_config(1, 17408, 5120)["BLOCK_N"] == 32
    finally:
        set_config_mode("gemv-bk512")
    with pytest.raises(ValueError):
        set_config_mode("auto")
    with pytest.raises(ValueError):
        choose_config(1, 5120, 5120, mode="auto")
    with pytest.raises(ValueError):
        validate_config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3})
    with pytest.raises(ValueError):
        validate_config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3})


def test_prefill_buckets_use_measured_family():
    from inference_from_scratch.fp8_triton import PREFILL_TUNED
    for m in (64, 65, 128, 512, 1024):
        for n, k in ((2560, 8704), (17408, 5120), (4096, 4096)):
            assert choose_config(m, n, k, mode="new") == PREFILL_TUNED
    assert PREFILL_TUNED == {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 128,
                             "num_warps": 8, "num_stages": 3}
    # Old mode never uses the prefill family; decode M=1 untouched.
    assert choose_config(128, 5120, 5120, mode="old")["BLOCK_N"] == 64
    assert choose_config(1, 5120, 17408, mode="new")["BLOCK_M"] == 16
    # Just below the prefill floor keeps legacy/new decode behaviour.
    assert choose_config(63, 5120, 5120, mode="new") == choose_config(63, 5120, 5120, mode="old")


def test_gemv_default_m1_only():
    from inference_from_scratch.fp8_triton import M1_GEMV_CONFIG, gemv_enabled
    assert M1_GEMV_CONFIG == {"BLOCK_N": 16, "BLOCK_K": 256,
                              "num_warps": 4, "num_stages": 3}
    # Selection: only mode "gemv" at exactly M=1 uses the GEMV kernel.
    assert gemv_enabled(1, mode="gemv") is True
    for m in (2, 8, 32, 64, 128):
        assert gemv_enabled(m, mode="gemv") is False
    for mode in ("new", "old"):
        assert gemv_enabled(1, mode=mode) is False
    with pytest.raises(ValueError):
        gemv_enabled(1, mode="auto")
    # Fallback: every (mode="gemv") dot selection is exactly "new".
    for m in (1, 2, 8, 32, 63, 64, 128, 512):
        for n, k in ((5120, 17408), (4096, 4096)):
            assert choose_config(m, n, k, mode="gemv") == choose_config(m, n, k, mode="new")
    # GEMV modes are shipped; round-4 default is gemv-bk512 (below).
    assert ft.config_mode() == "gemv-bk512"
    assert choose_config(1, 5120, 17408, mode="new")["BLOCK_N"] == 32
    try:
        assert set_config_mode("gemv") == "gemv"
        assert gemv_enabled(1) is True and gemv_enabled(4) is False
        assert choose_config(128, 5120, 5120)["BLOCK_M"] == 64
    finally:
        set_config_mode("gemv-bk512")
    with pytest.raises(ValueError):
        set_config_mode("splitn")


def test_gemv_bk512_default():
    from inference_from_scratch.fp8_triton import (
        M1_GEMV_BK512_CONFIG, M1_GEMV_CONFIG, gemv_config, gemv_enabled)
    assert M1_GEMV_BK512_CONFIG == {"BLOCK_N": 16, "BLOCK_K": 512,
                                    "num_warps": 4, "num_stages": 3}
    # Single variable vs the shipped GEMV config: only BLOCK_K.
    assert {k: v for k, v in M1_GEMV_BK512_CONFIG.items() if k != "BLOCK_K"} == \
        {k: v for k, v in M1_GEMV_CONFIG.items() if k != "BLOCK_K"}
    assert gemv_config(mode="gemv-bk512") == M1_GEMV_BK512_CONFIG
    assert gemv_config(mode="gemv") == M1_GEMV_CONFIG
    assert gemv_config(mode="new") == M1_GEMV_CONFIG
    # GEMV only at exactly M=1; M>1 dot selection is exactly "new".
    assert gemv_enabled(1, mode="gemv-bk512") is True
    for m in (2, 8, 32, 64, 128, 512):
        assert gemv_enabled(m, mode="gemv-bk512") is False
        assert choose_config(m, 5120, 17408, mode="gemv-bk512") == \
            choose_config(m, 5120, 17408, mode="new")
    # Round-4 promotion: gemv-bk512 is the shipped default.
    assert ft.config_mode() == "gemv-bk512"
    try:
        assert set_config_mode("gemv-bk512") == "gemv-bk512"
        assert gemv_enabled(1) is True and gemv_enabled(4) is False
        assert gemv_config() == M1_GEMV_BK512_CONFIG
    finally:
        set_config_mode("gemv-bk512")
    assert ft.config_mode() == "gemv-bk512"


def test_prefill_s2_rejected_removed():
    # Prefill round 1 rejected prefill-s2 (-1.3% to -7.5% everywhere);
    # the mode was removed, not kept as an opt-in.
    with pytest.raises(ValueError):
        set_config_mode("prefill-s2")
    with pytest.raises(ValueError):
        choose_config(512, 5120, 17408, mode="prefill-s2")
    with pytest.raises(ValueError):
        ft.gemv_enabled(1, mode="prefill-s2")
    assert ft.config_mode() == "gemv-bk512"
    from inference_from_scratch.fp8_triton import PREFILL_TUNED
    assert choose_config(512, 5120, 17408, mode="gemv-bk512") == PREFILL_TUNED


def test_existing_default_preserved_for_m_gt1():
    # choose_config contract from test_fp8_triton_dispatch still holds at M=128.
    assert choose_config(128, 5120, 5120, mode="new")["BLOCK_M"] == 64
    c = choose_config(1, 17408, 5120, mode="old")
    assert c["BLOCK_K"] == 128 and 128 % c["BLOCK_N"] == 0
