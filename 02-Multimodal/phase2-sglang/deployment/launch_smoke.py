"""Launch the fixed, loopback-only SGLang smoke configuration."""
import importlib.metadata
import json
import os
import sys
from pathlib import Path


def main(config, model_path):
    if importlib.metadata.version("sglang") != config["sglang_version"]:
        raise RuntimeError("The installed SGLang version does not match the smoke configuration")
    if not (model_path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("Qwen checkpoint index not found in local model path")
    args = [
        "--model-path", str(model_path.resolve()), "--revision", config["revision"],
        "--served-model-name", config["served_model_name"], "--host", "127.0.0.1",
        "--port", str(config["port"]), "--dtype", config["dtype"],
        "--tp-size", str(config["tp_size"]), "--context-length", str(config["context_length"]),
        "--mem-fraction-static", str(config["mem_fraction_static"]),
        "--max-running-requests", str(config["max_running_requests"]),
        "--attention-backend", config["attention_backend"],
        "--enable-metrics", "--mm-preprocess-cache-size-mb", "0",
    ]
    if config.get("radix_cache", False):
        args.extend(["--enable-cache-report", "--schedule-policy", "fcfs"])
    else:
        args.append("--disable-radix-cache")
    print(json.dumps({"launch_config": config, "package_versions": {
        name: importlib.metadata.version(name)
        for name in ("sglang", "torch", "transformers", "sglang-kernel")
    }}), flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "sglang.launch_server", *args])


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))
