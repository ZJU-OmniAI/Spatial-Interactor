# 3RSCAN Geometry-First Three-Task Builder

这个目录把 `/path/to/workspace/3RScan_sequence_only` 里的 `sequence.zip` 直接转换成和 `/path/to/workspace/DATA/REAL` 同风格的三类运动理解数据：

- `action_inference`
- `movement_sequence_sorting`
- `movement_degree_comparison`

输入依赖：

- `sequence.zip`
  - `frame-XXXXXX.color.jpg`
  - `frame-XXXXXX.pose.txt`
  - `_info.txt`

输出默认写到：

- `/path/to/workspace/DATA/3RSCAN`

主脚本：

- `build_3rscan_motion_qa.py`
  - 直接从 zip 中读 pose 和 RGB，生成三类样本目录、`qa_data.json`、`qa_data_all.json`、`summary.json`
- `prepare_3rscan_final_three_tasks.py`
  - 把原始中文 QA 整理成和 `REAL` 一致的 standardized / eval / prompt-pair 文件

建议流程：

```bash
python /path/to/workspace/DATA/CODE/SCENE/3RSCAN/build_3rscan_motion_qa.py
python /path/to/workspace/DATA/CODE/SCENE/3RSCAN/prepare_3rscan_final_three_tasks.py
```
