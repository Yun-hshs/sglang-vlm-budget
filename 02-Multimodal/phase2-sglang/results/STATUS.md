# 阶段 2 实验状态

**部署冒烟通过（PASS）。** 本地 9 项 CPU 测试通过。RTX 5090 单卡服务完成单图读数和双图差值，实测视觉 Token 为 2,691 和 5,382。[部署报告](../reports/DEPLOYMENT_SMOKE.md) · [请求记录](../reports/deployment-smoke.json)

**ChartQA 小样本预算对照完成（PASS）。** 32 道固定题、4 档、3 轮，共 384 条测量和 24 条预热，逐请求核对服务端视觉 Token。[结果、错例与复现](../reports/CHARTQA_BUDGET_PILOT.md)

**ChartQA 128 题五档对照完成（PASS）。** Human 与 augmented 各 64 题，加入自适应档，共 1,920 条测量与 30 条预热。阶段 1 Gradio 原进程已关闭，Demo 文件与权重保留。[五档结果与复现](../reports/CHARTQA_BUDGET_128.md)

**自适应规则改进对照完成（PASS）。** 同一批 128 题复跑旧规则、改进规则与固定中档，1,152 条测量与 21 条预热，阶段 1 Gradio 保持关闭。[结果、错例与复现](../reports/CHARTQA_ADAPTIVE_V2.md)

**独立图像留出集验证完成（PASS）。** 冻结规则在 128 张未见图上三组重复测量，1,152 条计量和 21 条预热。[结果、错例与复现](../reports/CHARTQA_ADAPTIVE_HOLDOUT.md)

**高档长图专项完成（PASS）。** 排除前两批图像后，纳入全部达门槛题与 5 张近高度门槛图，三策略重复 10 轮。[分组结果、错例与复现](../reports/CHARTQA_ADAPTIVE_HIGH_BRANCH.md)

**多轮追问代价拆解完成（PASS）。** 3 个并发会话、每会话 4 轮，固定中档做 4 次配对测量。缓存命中 Token、请求阶段时间及视觉 CUDA 批次逐条联接。[结果与复现](../reports/CHARTQA_SESSION_COST.md)

**Prefill 代价预测完成（PASS）。** 24 个完整会话按 15/9 划分，比较输入 Token、视觉与历史、派发前缓存估计三组特征。[结果与复现](../reports/CHARTQA_PREFILL_PREDICTION.md)

**最小调度对照完成（PASS）。** FCFS、原生 LPM、代价优先、代价优先加老化四组，各 108 条计量请求。[结果与复现](../reports/CHARTQA_SCHEDULE_COMPARE.md)

**持续负载对照完成（PASS）。** 54 个会话的连续提交轨迹，四种策略各完成 648 条计量请求，逐用户与长请求统计已保存。[结果与复现](../reports/CHARTQA_SCHEDULE_SUSTAINED.md)

**老化阈值消融完成（PASS）。** 100、220、350 ms 与 FCFS、纯代价优先在同一持续负载轨迹上各完成 648 条计量请求。[实测与开源方案](../reports/CHARTQA_SCHEDULE_AGING_ABLATION.md)

**本地 Prefill 校准与原生 SPF 对照完成（PASS）。** 校准和验证图像及会话独立。验证探针计量 288 条，线上四组各计量 432 条。[实测与复现](../reports/CHARTQA_PREFILL_SIGNALS.md)

**hybrid 多用户 Web 与故障恢复完成（PASS）。** 固定 HTTPS 入口实测通过登录、隔离历史、流式回答、整套进程重启及组件故障恢复。[部署与验收](../reports/VISUAL_CHAT_MVP.md)

**正式预算实验待运行（NOT_RUN）。** 图数、并发、预算与质量的对照矩阵以及缓存扩展仍待执行。

[实验登记表](experiment_register.csv) · [阶段 2 复现入口](../README.md)

后续运行须记录模型版本、原始请求、样本清单与哈希、运行环境和指标定义。已完成的 ChartQA 基准和 Gradio 验收位于[基础作业正式报告](../../phase1-assistant/reports/final-report.md)。
