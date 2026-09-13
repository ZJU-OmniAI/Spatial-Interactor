# Official Pose Rules

统一目标格式：

- OpenCV / COLMAP convention
- `world-to-camera` (`w2c`) extrinsics

## ARKitScenes

官方文件：

- `lowres_wide.traj`

当前规则：

- 每行解析为：
  - `timestamp angle_axis_x angle_axis_y angle_axis_z tx ty tz`
- 先把 angle-axis 转成旋转矩阵 `R`
- 组装矩阵：
  - `T = [[R, t], [0, 0, 0, 1]]`
- 当前 probe 结果表明，应将该矩阵**直接视为 `w2c`**

当前本机验证：

- `41069025`
- `41069043`

输出见：

- `/path/to/workspace/SCENEOUTPUT/REAL/pose_probe/arkitscenes/`

## ScanNet

官方文件：

- 每帧 pose txt

规则：

- 官方 pose 通常表示 `camera-to-world` (`c2w`)
- 统一到本项目时使用：
  - `w2c = inv(c2w)`

当前状态：

- 本机暂未发现对应场景的官方 pose txt
- `pose_rule_probe.py` 已支持该格式，补齐官方 pose 后可直接验证

## ScanNet++

可能遇到三种文件：

1. `pose_intrinsic_imu.json`
- 规则：
  - iPhone pose 视为 `c2w`
  - `w2c = inv(c2w)`

2. `transforms.json`
- 规则：
  - Nerfstudio / OpenGL 风格 `c2w`
  - 先做 OpenGL -> OpenCV 轴转换
  - 再取逆得到 `w2c`

3. `images.txt`
- 规则：
  - COLMAP 外参本身就是 `w2c`
  - 直接使用

当前状态：

- 本机暂未发现对应场景的官方 pose 文件
- `pose_rule_probe.py` 已支持上述三种格式，补齐后可直接 probe
