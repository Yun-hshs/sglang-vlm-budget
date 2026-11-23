"""Input pixel budgets, NOT measurements of actual vision tokens."""
import base64
import io
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from PIL import Image, ImageOps

BUDGETS = {"low": 256*28*28, "medium": 768*28*28, "high": 1280*28*28}

@dataclass(frozen=True)
class PreparedImage:
    url: str
    metadata: dict

def prepare_image(path, strategy="baseline", question=""):
    if strategy not in {"baseline", "adaptive", *BUDGETS}:
        raise ValueError("Unknown budget strategy")
    path = Path(path)
    if path.stat().st_size > 25*1024*1024:
        raise ValueError("图片文件不得超过 25 MiB")
    with Image.open(path) as source:
        if source.width*source.height > 40_000_000:
            raise ValueError("图片不得超过 4000 万像素")
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
    original = image.size
    chosen = strategy
    if strategy == "adaptive":
        # No answer labels or dataset category used. Decision cost is in preprocessing time.
        text_hint = any(k in question.lower() for k in
            ["文字", "文档", "图表", "读取", "写", "数字", "text", "read", "document", "chart", "ocr"])
        portrait = image.height/image.width > 1.35 and image.height > 1600
        chosen = "high" if text_hint or portrait else "low"
    cap = BUDGETS.get(chosen)
    if cap and image.width*image.height > cap:
        scale = math.sqrt(cap/(image.width*image.height))
        image = image.resize((max(1,int(image.width*scale)), max(1,int(image.height*scale))),Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return PreparedImage("data:image/png;base64,"+base64.b64encode(output.getvalue()).decode(), {
        "strategy":strategy,"selected_budget":chosen,"pixel_cap":cap,
        "original_width":original[0],"original_height":original[1],
        "width":image.width,"height":image.height,
        "visual_tokens_actual":None,
        "token_note":"Actual tokens require server image_grid_thw and spatial merge instrumentation.",
    })
