"""Install a version-specific, local-only observation hook in the SGLang wheel."""
import argparse
from importlib.metadata import distributions
from pathlib import Path

VERSION = "0.5.23"
PROCESSOR = Path("sglang/srt/multimodal/processors/qwen_vl.py")
ORIGINAL = "    async def process_mm_data_async("
PATCH = '''    async def process_mm_data_async(self, image_data, input_text, request_obj, *args, **kwargs):
        output = await self._stage2_original_process_mm_data_async(
            image_data, input_text, request_obj, *args, **kwargs
        )
        from deployment.mm_trace import emit_visual_trace
        emit_visual_trace(self, request_obj, output)
        return output

    async def _stage2_original_process_mm_data_async('''


def install(runtime):
    packages = {dist.metadata["Name"]: dist.version for dist in distributions(path=[str(runtime)])}
    if packages.get("sglang") != VERSION:
        raise ValueError(f"The isolated runtime must contain SGLang {VERSION}")
    processor = runtime / PROCESSOR
    code = processor.read_text()
    if "    async def _stage2_original_process_mm_data_async(" in code:
        return
    if code.count(ORIGINAL) != 1:
        raise ValueError("Unexpected Qwen-VL processor signature")
    updated = code.replace(ORIGINAL, PATCH, 1)
    compile(updated, str(processor), "exec")
    processor.write_text(updated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    install(parser.parse_args().runtime.resolve())
