"""Record real Qwen-VL processor outputs for explicitly marked smoke requests."""
import json
import os


def emit_visual_trace(processor, request, output):
    target = os.getenv("SGLANG_MM_TRACE_JSONL")
    request_id = getattr(request, "rid", None)
    if not target or not isinstance(request_id, str) or not request_id.startswith("stage2-mm-"):
        return

    merge = int(processor.spatial_merge_size)
    ids = output.input_ids
    image_id = output.im_token_id
    images = []
    for item in output.mm_items:
        if item.modality.name != "IMAGE":
            continue
        grid = item.model_specific_data["image_grid_thw"]
        rows = grid.reshape(-1, 3).tolist()
        for (t, h, w), (start, end) in zip(rows, item.offsets, strict=True):
            if h % merge or w % merge:
                raise ValueError("Unaligned Qwen-VL visual grid")
            t, h, w, start, end = map(int, (t, h, w, start, end))
            images.append({
                "grid_thw": [t, h, w],
                "merged_tokens": t * (h // merge) * (w // merge),
                "offset": [start, end],
                "offset_tokens": end - start + 1,
                "image_pad_tokens": sum(int(token) == image_id for token in ids[start:end + 1]),
            })
    event = {
        "request_id": request_id,
        "spatial_merge_size": merge,
        "images": images,
        "visual_tokens": sum(image["merged_tokens"] for image in images),
        "prompt_image_pad_tokens": sum(int(token) == image_id for token in ids),
        "processor_prompt_tokens": len(ids),
    }
    line = (json.dumps(event, ensure_ascii=False) + "\n").encode()
    descriptor = os.open(target, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, line)
    finally:
        os.close(descriptor)
