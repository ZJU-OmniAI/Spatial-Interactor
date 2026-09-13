# AI2THOR Scene-Exhaustive Framework

第一版目标不是直接生成最终 QA，而是先把固定 120 个房间的穷举框架搭起来：

1. 固定 120 个 AI2THOR 房间。
2. 枚举每个房间的 reachable positions / yaw / horizon。
3. 构建高质量初始状态库 `state_bank.jsonl`。
4. 基于初始状态库，为 10 类问题生成结构化 `proposal` 清单。
5. 后续再接正式渲染和 QA 验证。

## 当前文件

- `scene_catalog.py`
  - 固定 120 个房间列表。
- `state_quality.py`
  - 初始状态质量打分、可见物体摘要、类适配性判定。
- `build_state_bank.py`
  - 构建单场景或多场景状态库。
- `scene_worker.py`
  - 单场景完整流水线：状态库 + proposal 枚举。
- `run_parallel.py`
  - 多进程并行跑多个场景。
- `enumerators/`
  - 10 类 proposal 枚举器接口和第一版启发式枚举。

## 当前阶段已完成

- 120 房间固定列表。
- 场景级状态枚举。
- 初始状态质量筛选。
- 10 类 proposal 框架接口。
- 多进程场景级并行。
- 第1类已经从“随机占位 proposal”升级成“按状态穷举动作模板 proposal”。
- 第1类已经新增 proposal 执行层，可把 class01 proposal bank 真正执行成图像与 QA。

## 当前阶段尚未完成

- proposal 到正式图像/问答渲染的执行层。
- 第 1 类到第 10 类的“完全穷举式细粒度枚举器”。
- 第 6 类等需要和最终 QA 结果严格一致的硬验证执行器。
- 最终结果合并与 HTML 预览生成。

## 用法

单场景：

```bash
/path/to/workspace/SCENE/AI2THOR/run_scene_worker.sh \
  --scenes FloorPlan1 \
  --output-root /path/to/workspace/SCENE/AI2THOR/output
```

并行全场景：

```bash
/path/to/workspace/SCENE/AI2THOR/run_parallel.sh \
  --output-root /path/to/workspace/SCENE/AI2THOR/output \
  --num-workers 8
```

按房间类型各选 5 个场景，只生成 `state_bank + proposals`，不渲染：

```bash
/path/to/workspace/SCENE/AI2THOR/run_roomtype_subset.sh \
  --output-root /path/to/workspace/SCENE/AI2THOR/output_roomtype_subset \
  --scenes-per-type 5 \
  --num-workers 4
```

默认场景选择：

- `kitchen`: `FloorPlan1-5`
- `living_room`: `FloorPlan201-205`
- `bedroom`: `FloorPlan301-305`
- `bathroom`: `FloorPlan401-405`

默认严格筛选参数：

- `horizon=0`
- 每个可达点最多 `2` 个朝向
- 单场景最多 `50` 个初始状态
- 可达点最小间距 `0.75m`

跑完后会生成：

- 每个场景自己的 `state_bank.jsonl` 和 `proposals.jsonl`
- 总汇总文件 `roomtype_subset_overview.json`

执行第1类 proposal bank：

```bash
/path/to/workspace/SCENE/AI2THOR/run_execute_class01.sh \
  --state-bank /path/to/workspace/SCENE/AI2THOR/output/FloorPlan1/state_bank.jsonl \
  --proposals /path/to/workspace/SCENE/AI2THOR/output/FloorPlan1/proposals.jsonl \
  --output-root /path/to/workspace/SCENE/AI2THOR/output/FloorPlan1/class01_rendered \
  --subcats two_image_single_action \
  --max-proposals 50
```
