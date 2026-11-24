"""Input pixel budgets, NOT measurements of actual vision tokens."""
import base64
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageOps

BUDGETS = {"low": 256*28*28, "medium": 768*28*28, "high": 1280*28*28}
TEXT_QUESTION = re.compile(r"\b(?:ocr|text|read|document)\b", re.IGNORECASE)
COMPARISON_QUESTION = re.compile(
    r"\b(?:differences?|averages?|means?|sums?|totals?|aggregate[ds]?|median|mode|ratio|"
    r"combined|altogether|together|compare[ds]?|comparison|versus|vs|both|"
    r"subtract(?:ed)?|add(?:ed)?|percentage points?|smaller|larger|"
    r"how\s+(?:much|many)\s+(?:more|less|fewer|higher|lower)|by\s+how\s+much|"
    r"(?:more|less|fewer|higher|lower|greater)\s+than|"
    r"(?:more|less|fewer|higher|lower)\s+\w+\s+than)\b",
    re.IGNORECASE,
)
CHART_REASONING_HINTS = ("差值", "相差", "平均", "总和", "合计", "聚合", "中位数", "众数", "相比", "比较", "相加", "百分点")
TEXT_HINTS = ("文字", "文档", "读取")

@dataclass(frozen=True)
class PreparedImage:
    url: str
    metadata: dict

def prepare_image(path, strategy="baseline", question=""):
    if strategy not in {"baseline", *ADAPTIVE_RULES, *BUDGETS}:
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
    chosen = select_budget(strategy, image.width, image.height, question)
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


def adaptive_budget(width, height, question):
    text_hint = any(word in question.lower() for word in (
        "文字", "文档", "图表", "读取", "写", "数字", "text", "read", "document", "chart", "ocr",
    ))
    portrait = height / width > 1.35 and height > 1600
    return "high" if text_hint or portrait else "low"


def adaptive_budget_v2(width, height, question):
    # Area and aspect ratio approximate dense long charts using only image dimensions.
    if height > 1600 and height / width > 1.35 and width * height > BUDGETS["high"]:
        return "high"
    if (
        COMPARISON_QUESTION.search(question) or TEXT_QUESTION.search(question)
        or any(hint in question for hint in (*CHART_REASONING_HINTS, *TEXT_HINTS))
    ):
        return "medium"
    return "low"


ADAPTIVE_RULES = {"adaptive": adaptive_budget, "adaptive_v2": adaptive_budget_v2}


def select_budget(strategy, width, height, question):
    if strategy in ADAPTIVE_RULES:
        return ADAPTIVE_RULES[strategy](width, height, question)
    return strategy
