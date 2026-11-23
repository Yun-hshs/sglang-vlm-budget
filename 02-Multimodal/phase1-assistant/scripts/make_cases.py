"""Create original image/video fixtures for functional acceptance."""
import argparse
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw


def chart(path, apple, pear):
    image = Image.new("RGB", (1920, 1080), "#f4f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((180, 80), "FRUIT SALES (UNITS)", fill="#121e30", font_size=70)
    for y, label, value in [(340, "Apples", apple), (640, "Pears", pear)]:
        draw.text((180, y), label, fill="#121e30", font_size=62)
        endpoint = 560 + value * 10
        draw.rectangle((560, y, endpoint, y + 100), fill="#2355a6")
        draw.text((endpoint + 24, y + 20), str(value), fill="#121e30", font_size=58)
    image.save(path)


def make_cases(output):
    output.mkdir(parents=True, exist_ok=False)
    chart(output / "day1.png", 40, 20)
    chart(output / "day2.png", 70, 25)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=red:s=384x256:r=12:d=2",
         "-f", "lavfi", "-i", "color=blue:s=384x256:r=12:d=2",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(output / "red-blue.mp4")], check=True,
    )
    cases = [
        {"id": "hd-chart", "subset": "hd", "images": ["day1.png"],
         "question": "How many units of apples were sold? Reply with the number only.", "answers": ["40"]},
        {"id": "across-charts", "subset": "multi-image", "images": ["day1.png", "day2.png"],
         "question": "How many more apple units are sold in Image 2 than in Image 1? Reply with the number only.",
         "answers": ["30"]},
        {"id": "video-order", "subset": "video", "video": "red-blue.mp4",
         "question": "Which color fills the screen first, red or blue? Answer with one color word.",
         "answers": ["red"]},
    ]
    (output / "cases.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    make_cases(parser.parse_args().output)
