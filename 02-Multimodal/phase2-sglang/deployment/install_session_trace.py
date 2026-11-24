"""Install request and batch timing hooks in the isolated SGLang 0.5.23 wheel."""
import argparse
from pathlib import Path

from deployment.install_trace import PATCH, install


def replace(path, original, updated):
    code = path.read_text()
    if updated in code:
        return
    if code.count(original) != 1:
        raise ValueError(f"Unexpected tracing anchor in {path}")
    patched = code.replace(original, updated, 1)
    compile(patched, str(path), "exec")
    path.write_text(patched)


def install_session(runtime):
    install(runtime)
    srt = runtime / "sglang/srt"
    anchor = "                del self.rid_to_state[rid]\n\n                self._release_lora_once(state)"
    replace(srt / "managers/tokenizer_manager.py", anchor,
            '                if rid.startswith("stage2-mm-session-"):\n'
            '                    from deployment.session_trace import emit_runtime_trace\n'
            '                    emit_runtime_trace(rid, meta_info)\n\n' + anchor)

    timed_processor = PATCH.replace(
        "        output = await", "        from time import perf_counter\n        started_at = perf_counter()\n        output = await", 1,
    ).replace(
        "        from deployment.mm_trace import emit_visual_trace",
        "        elapsed = perf_counter() - started_at\n"
        "        from deployment.session_trace import emit_processor_trace\n"
        "        emit_processor_trace(request_obj, elapsed)\n"
        "        from deployment.mm_trace import emit_visual_trace",
        1,
    )
    replace(srt / "multimodal/processors/qwen_vl.py", PATCH, timed_processor)

    anchor = "    input_embeds = prepare_mm_inputs(\n"
    replace(srt / "managers/mm_utils.py", anchor,
            "    from deployment.session_trace import timed_mm_preparation\n"
            "    input_embeds = timed_mm_preparation(prepare_mm_inputs,\n")
    anchor = "            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)"
    replace(srt / "models/qwen2_5_vl.py", anchor,
            "            from deployment.session_trace import timed_vision_forward\n"
            "            image_embeds = timed_vision_forward(self.visual, pixel_values, grid_thw=image_grid_thw)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    install_session(parser.parse_args().runtime.resolve())
