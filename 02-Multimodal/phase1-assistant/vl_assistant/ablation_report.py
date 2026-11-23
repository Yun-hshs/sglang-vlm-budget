"""Create a paired ChartQA scoreboard and representative observed error cases."""
import argparse
import json
from pathlib import Path

from .scoring import relaxed_correct


def report(root):
    def load(relative):
        return json.loads((root / relative).read_text())

    selection = load("selection.json")
    summaries = {"基座": load("test/base/summary.json"),
                 **{f"rank {rank}": load(f"test/rank-{rank}/summary.json") for rank in selection}}
    lines = ["# ChartQA 全量测试", "",
             "| 模型 | Human（1,250） | Augmented（1,250） | 合计（2,500） |",
             "|---|---:|---:|---:|"]
    for name, summary in summaries.items():
        if any(summary["scores"][group]["n"] != n or summary["scores"][group]["failed"]
               for group, n in (("human", 1250), ("augmented", 1250), ("overall", 2500))):
            raise ValueError(f"Incomplete full test results for {name}")
        cells = [f"{s['correct']}/{s['n']} ({s['relaxed_accuracy']:.2%})"
                 for group in ("human", "augmented", "overall") if (s := summary["scores"][group])]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 验证集选点", "",
                  "按 128 条固定验证样本的总正确数选点。同分时选择训练步数较早的 checkpoint。", ""])
    for rank, value in selection.items():
        candidates = "，".join(f"step {c['step']} 为 {c['scores']['overall']['correct']}/128"
                               for c in value["candidates"])
        lines.append(f"- rank {rank}：{candidates}。选择 step {value['selected']['step']}。")

    test = {row["id"]: row for row in
            map(json.loads, (root / "test/base/predictions.jsonl").read_text().splitlines())}
    comparisons = {rank: load(f"comparison-rank-{rank}.json") for rank in selection}
    lines.extend(["", "## 错误案例与变化", ""])
    examples = []
    for rank, result in comparisons.items():
        paired = result["paired"]
        lines.append(f"- rank {rank}：纠正 {paired['improved']} 题，退化 {paired['regressed']} 题，"
                     f"输出文本变化 {paired['changed_prediction']} 题。")
        for subset in ("human", "augmented"):
            group = [row for row in result["rows"] if row["subset"] == subset]
            for label, predicate in (
                ("纠正", lambda row: not row["base_correct"] and row["lora_correct"]),
                ("退化", lambda row: row["base_correct"] and not row["lora_correct"]),
                ("持续错误", lambda row: not row["base_correct"] and not row["lora_correct"]),
            ):
                row = next((row for row in group if predicate(row)), None)
                if row:
                    original = test[row["id"]]
                    examples.append({"rank": rank, "case": label, **row,
                                     "question": original["question"], "answers": original["answers"],
                                     "image": original["image"]})
    for item in examples:
        lines.extend(["", f"- **rank {item['rank']} · {item['subset']} · {item['case']}**",
                      f"  - 样本 {item['id']}。{item['question']}",
                      f"  - 参考答案 {item['answers']}，基座 `{item['base_prediction']}`，"
                      f"LoRA `{item['lora_prediction']}`。"])
    format_counts = {}
    for name, folder in [("base", "base"), *((rank, f"rank-{rank}") for rank in selection)]:
        rows = [json.loads(line) for line in
                (root / "test" / folder / "predictions.jsonl").read_text().splitlines()]
        format_counts[name] = {
            subset: sum(row["subset"] == subset and not row["correct"]
                        and row["prediction"].endswith("%")
                        and all(not answer.endswith("%") for answer in row["answers"])
                        and relaxed_correct(row["prediction"][:-1], row["answers"])
                        for row in rows)
            for subset in ("human", "augmented")
        }
    lines.extend(["", "### 百分号格式诊断", "",
                  "以下统计只检查已判错的题目。预测末尾带百分号、参考答案没有百分号，"
                  "且去掉百分号后数值落入 5% 容差的题数如下。正式分数沿用原评分规则。", "",
                  "| 模型 | Human | Augmented |",
                  "|---|---:|---:|"])
    for name, counts in format_counts.items():
        lines.append(f"| {name} | {counts['human']} | {counts['augmented']} |")
    (root / "format-diagnostic.json").write_text(json.dumps(format_counts, indent=2) + "\n")
    (root / "results.md").write_text("\n".join(lines) + "\n")
    (root / "errors.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n")
    print("\n".join(lines[:10]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    report(parser.parse_args().run)
