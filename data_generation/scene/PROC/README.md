# ProcTHOR Scene-Exhaustive Framework

第一版目标不是直接生成最终 QA，而是先把固定种子 ProcTHOR 房间的穷举框架搭起来：

1. 固定一批 ProcTHOR house seed。
2. 枚举每个房间的 reachable positions / yaw / horizon。
3. 构建高质量初始状态库 `state_bank.jsonl`。
4. 基于初始状态库，为 10 类问题生成结构化 `proposal` 清单。
5. 后续再接正式渲染和 QA 验证。

## 当前文件

- `scene_catalog.py`
  - 固定 ProcTHOR seed 列表。
- `state_quality.py`
  - 初始状态质量打分、可见物体摘要、类适配性判定。
- `build_state_bank.py`
  - 构建单场景或多场景状态库。
- `scene_worker.py`
  - 单场景完整流水线：状态库 + proposal 枚举。
- `run_parallel.py`
  - 多进程并行跑多个 ProcTHOR seed。
- `run_scene_subset.py`
  - 按 offset/limit 跑固定 seed 子集。
- `enumerators/`
  - 10 类 proposal 枚举器接口和第一版启发式枚举。

## 当前阶段已完成

- ProcTHOR seed 固定列表。
- 场景级状态枚举。
- 初始状态质量筛选。
- 10 类 proposal 框架接口。
- 多进程场景级并行。
- 第1-10类 proposal 执行入口已经补齐。
- trytest 采样和预览入口已经补齐。

## 当前阶段尚未完成

- 全量 smoke 运行验证。
- 更细的 repair 工作流还需要结合实际失败小类再补。
- 最终结果质量还要继续针对小类逐个收紧。

## 用法

单场景：

```bash
/path/to/workspace/SCENE/PROC/run_scene_worker.sh \
  --scenes ProcTHORTrain1 \
  --output-root /path/to/workspace/SCENE/PROC/output
```

并行全 seed：

```bash
/path/to/workspace/SCENE/PROC/run_parallel.sh \
  --output-root /path/to/workspace/SCENE/PROC/output \
  --num-workers 8
```

按子集跑，只生成 `state_bank + proposals`，不渲染：

```bash
/path/to/workspace/SCENE/PROC/run_scene_subset.sh \
  --output-root /path/to/workspace/SCENE/PROC/output_subset \
  --offset 0 \
  --limit 20 \
  --num-workers 4
```

默认严格筛选参数：

- `horizon=0`
- 每个可达点最多 `2` 个朝向
- 单场景最多 `50` 个初始状态
- 可达点最小间距 `0.75m`

跑完后会生成每个 seed 自己的 `state_bank.jsonl` 和 `proposals.jsonl`，以及总汇总文件。

执行第1类 proposal bank：

```bash
/path/to/workspace/SCENE/PROC/run_execute_class01.sh \
  --state-bank /path/to/workspace/SCENE/PROC/output/ProcTHORTrain1/state_bank.jsonl \
  --proposals /path/to/workspace/SCENE/PROC/output/ProcTHORTrain1/proposals.jsonl \
  --output-root /path/to/workspace/SCENE/PROC/output/ProcTHORTrain1/class01_rendered \
  --subcats two_image_single_action \
  --max-proposals 50
```
