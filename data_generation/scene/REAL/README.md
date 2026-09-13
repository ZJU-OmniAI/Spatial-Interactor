# REAL Geometry-First Prototype

这个目录现在切到几何优先路线：不再把已有 QA 当监督，而是先统一三类真实数据的官方 pose 转换规则，再基于官方 pose / 深度 / 重建去生成真值。

第一步目标不是直接出最终 QA，而是先把下面这件事做稳：

- `official pose -> 统一的 OpenCV/COLMAP world-to-camera extrinsics`

统一后，才能继续做真实视频里第一阶段的运动类 QA。

## 当前参考系

- `DA3` 输出作为参考坐标系
- 目标统一格式：
  - `w2c` extrinsics
  - OpenCV / COLMAP 约定

`camera-trans` 和 `Depth-Anything-3/workspace/vsibench_pose_cache` 已经提供了大量 DA3 结果，可直接拿来和官方 pose 做 rule probe。

## 当前脚本

- `pose_rule_probe.py`
  - 自动读取官方 pose，枚举候选转换规则，对齐 DA3 参考轨迹并打分，输出最优规则。
- `generate_pose_gt_samples.py`
  - 用 `vsibench` 现成视频和 pose 资产直接生成一批小样本 QA。
  - 当前先覆盖纯位姿可监督的 3 类：
    - `action_inference`
    - `movement_sequence_sorting`
    - `movement_degree_comparison`
- `extract_aligned_real_frames.py`
  - 从 `VSI` 视频中抽取与官方 pose / intrinsics 对齐的真实帧。
  - `ScanNet` 使用严格 `frame_id == pose_id`。
  - `ARKitScenes` 使用官方时间戳对齐，并在残差超过阈值时拒绝输出。
- `run_extract_aligned_real_frames_batch.py`
  - 批量抽取对齐帧。
- `generate_motion_qa_from_aligned_frames.py`
  - 直接消费 `aligned_frames/*/metadata.json` 生成几何监督的真实视频 QA。
  - 当前覆盖 6 类：
    - `action_inference`
    - `movement_sequence_sorting`
    - `movement_degree_comparison`
    - `motion_family_discrimination`
    - `distance_to_start_comparison`
    - `return_to_start_detection`
- `real_phase1_lib.py`
  - 公共工具，暂时保留。
- `prepare_phase1_manifest.py`
  - 旧原型脚本，仍可保留做目录清点，但不再是主流程。
- `generate_phase1_samples.py`
  - 旧原型脚本，基于现成 QA 监督；现在不作为正式方向。

## 当前已支持的官方 pose 格式

- `arkitscenes_traj`
  - `lowres_wide.traj`
- `scannet_pose_dir`
  - 每帧一个 4x4 pose txt
- `scannetpp_pose_intrinsic_imu`
  - `pose_intrinsic_imu.json`
- `scannetpp_transforms_json`
  - `transforms.json`
- `scannetpp_colmap_images`
  - `images.txt`

## 当前输出

- probe 输出：
  - `/path/to/workspace/SCENEOUTPUT/REAL/pose_probe/`
- 小样本 QA：
  - `/path/to/workspace/SCENEOUTPUT/REAL/pose_gt_samples/`
- 对齐帧：
  - `/path/to/workspace/SCENEOUTPUT/REAL/aligned_frames/`
- 基于对齐帧生成的运动 QA：
  - `/path/to/workspace/SCENEOUTPUT/REAL/motion_qa/`

## 运行示例

ARKitScenes：

```bash
python3 /path/to/workspace/SCENE/REAL/pose_rule_probe.py \
  --dataset arkitscenes \
  --scene 41069025
```

如果后续补齐了本地官方 ScanNet / ScanNet++ pose 文件，也可以直接 probe：

```bash
python3 /path/to/workspace/SCENE/REAL/pose_rule_probe.py \
  --dataset scannet \
  --scene scene0608_00 \
  --official-path /path/to/pose_dir
```

先生成一批基于位姿 GT 的 QA：

```bash
python3 /path/to/workspace/SCENE/REAL/generate_pose_gt_samples.py
```

先批量抽真实视频对齐帧：

```bash
bash /path/to/workspace/SCENE/REAL/start_extract_aligned_real_frames_screen.sh
```

再用这些对齐帧生成 6 类运动 QA：

```bash
python3 /path/to/workspace/SCENE/REAL/generate_motion_qa_from_aligned_frames.py \
  --aligned-root /path/to/workspace/SCENEOUTPUT/REAL/aligned_frames \
  --output-root /path/to/workspace/SCENEOUTPUT/REAL/motion_qa
```

## 说明

- 当前本机能直接验证的是：
  - `ARKitScenes` 官方 `traj`
  - DA3 参考缓存
- `ScanNet / ScanNet++` 的 probe 逻辑已经实现，但是否能立刻跑通取决于本机有没有官方 pose 文件。
