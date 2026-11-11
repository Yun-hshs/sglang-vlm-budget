# 数据契约与合法来源

未下载任何数据集。以下是格式说明，不是实际样本或测量结果：

```json
{"id":"sample-001","images":["images/001.jpg"],"question":"图中有几只猫？只回答数字。","answers":["2"],"task":"general-vqa","metric":"em"}
```

images 相对于 manifest 文件目录，不从网络自动抓图。metric 可选 em / vqa / anls；vqa 必须提供 10 个原始人工答案。只有许可允许且本地图片已就绪的数据才能进入实验。

建议来源：

- VQAv2：https://visualqa.org/download.html；使用含公开答案的验证分割，按官网条款获取，COCO 图片许可另计。
- TextVQA：https://textvqa.org/dataset/；原始数据和图像获取条件以官方发布为准。
- DocVQA：https://www.docvqa.org/datasets/docvqa；可能需要注册并接受使用协议；不得擅自分发。

发布和使用条件仍需下载当天核实。当前未接受任何数据协议，也未将未授权内容上传 GitHub。

步骤：下载获准数据 → 转成上面 JSONL → 按任务各自固定种子抽样 → 保存来源、split、官方版本、许可、总量和 subset IDs → 跨策略复用。

建议每任务至少 100 样本，small smoke 子集 5 样本。多图压力实验应独立制作真实多图题与参考答案，不能把单图样本直接复制成多图质量评测。

正式结论使用官方 evaluator。本项目 normalize 是简化实现，英文缩写、标点与数字的处理不完全等同官方脚本。数据抽样脚本转为本地绝对路径，跨主机需重新生成 manifest；数据哈希用于核对内容而非路径一致性。
