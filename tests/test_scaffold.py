from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MODULES = ["config","tokenizer","sampling","layers","kv_cache","attention","mlp","model","generate","weights","quantization","serving"]
def test_modules_exist():
    for m in MODULES:
        assert (ROOT / "src/inference_from_scratch" / f"{m}.py").exists(), m
def test_package_version():
    import inference_from_scratch
    assert inference_from_scratch.__version__
