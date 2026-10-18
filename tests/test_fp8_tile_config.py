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
        set_config_mode("new")
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


def test_existing_default_preserved_for_m_gt1():
    # choose_config contract from test_fp8_triton_dispatch still holds at M=128.
    assert choose_config(128, 5120, 5120, mode="new")["BLOCK_M"] == 64
    c = choose_config(1, 17408, 5120, mode="old")
    assert c["BLOCK_K"] == 128 and 128 % c["BLOCK_N"] == 0
