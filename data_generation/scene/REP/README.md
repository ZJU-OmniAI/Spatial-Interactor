# ReplicaCAD Scene-Exhaustive Framework

这一层对齐 `/path/to/workspace/SCENE/AI2THOR` 的总体思路，但底层改为 Habitat/ReplicaCAD：

1. 固定 ReplicaCAD 场景列表。
2. 基于 navmesh 顶点枚举可导航起始点。
3. 为每个起始点筛选少量高质量朝向，生成 `state_bank.jsonl`。
4. 在 `state_bank` 上跑 10 类 proposal enumerators，生成 `proposals.jsonl`。

当前这版已经补到完整 pipeline 骨架：

1. `state_bank`
2. `proposal bank`
3. `execute_class01-10`
4. `trytest`
5. `preview html`

## 当前文件

- `scene_catalog.py`
  - 发现并固定本地 ReplicaCAD 场景。
- `hssd_env.py`
  - 复用 `habitat_qa_generators.py` 的 ReplicaCAD/Habitat 场景加载接口。
- `state_quality.py`
  - ReplicaCAD 视角质量评分、可见物体摘要、类适配性判定。
- `build_state_bank.py`
  - 构建单场景或多场景状态库。
- `scene_worker.py`
  - 单场景完整流水线：状态库 + proposal 枚举。
- `run_parallel.py`
  - 多进程并行跑多个场景。
- `run_scene_subset.py`
  - 跑固定场景子集。
- `bank_task_executor.py`
  - 将 proposal bank 执行成最终 QA 样本。
- `execute_class01_bank.py` 到 `execute_class10_bank.py`
  - 10 类执行入口。
- `enumerators/`
  - 复用 AI2THOR 第一版 10 类 proposal 枚举器。
- `trytest/`
  - 小类级采样执行、repair merge、preview。

## 用法

单场景：

```bash
/path/to/workspace/SCENE/REP/run_scene_worker.sh \
  --scenes apt_0 \
  --output-root /path/to/workspace/SCENE/REP/output
```

并行多场景：

```bash
/path/to/workspace/SCENE/REP/run_parallel.sh \
  --output-root /path/to/workspace/SCENE/REP/output \
  --num-workers 2
```

快速跑一个固定子集：

```bash
/path/to/workspace/SCENE/REP/run_scene_subset.sh \
  --output-root /path/to/workspace/SCENE/REP/output_subset \
  --limit 20 \
  --num-workers 2
```

执行某一类 proposal bank：

```bash
/path/to/workspace/SCENE/REP/run_execute_class01.sh \
  --state-bank /path/to/workspace/SCENE/REP/output/apt_0/state_bank.jsonl \
  --proposals /path/to/workspace/SCENE/REP/output/apt_0/proposals.jsonl \
  --output-root /path/to/workspace/SCENE/REP/output/apt_0/class01_rendered \
  --subcats two_image_single_action \
  --max-proposals 50
```

trytest 小类采样：

```bash
/path/to/workspace/SCENE/REP/trytest/run_trytest_subcats.sh --reset-output
```

建议先用保守参数 smoke：

```bash
/path/to/workspace/SCENE/REP/run_scene_worker.sh \
  --scenes apt_0 \
  --output-root /path/to/workspace/SCENE/REP/_smoke \
  --max-positions 40 \
  --max-total-states 20
```

并发建议：

- ReplicaCAD 先从 `--num-workers 1` 或 `2` 开始。
- 稳定后再逐步加大，不要一开始就按 AI2THOR 的并发数硬推。
