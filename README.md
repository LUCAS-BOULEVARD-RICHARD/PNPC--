# PNPC++

基于 OpenCV、yaml-cpp 的实时目标检测与 PnP 位姿解算工程。支持 USB/笔记本相机与海康工业相机，推理后端可在 TensorRT、LibTorch、OpenCV DNN ONNX 之间选择。

## 功能

- 实时 YOLO 目标检测
- 单目标 PnP 距离解算
- 目标位姿与欧拉角输出
- 完整可视化：检测框、坐标轴、HUD、目标面板
- 相机标定
- 独立曝光/增益调试工具
- 采集线程与处理线程分离，处理不过来时自动丢旧帧

## 目录结构

```text
PNPC++
|-- CMakeLists.txt              # CMake 构建配置
|-- config/
|   |-- config.yaml             # 主运行配置
|   `-- calibration.yaml        # 相机标定输出
|-- model/
|   |-- labubu_yolov8n.engine   # TensorRT 模型
|   `-- labubu_yolov8n.onnx     # ONNX 模型
|-- include/
|   |-- yaml.hpp                # YAML ConfigManager 封装
|   `-- pnp_vision/
|       |-- camera/             # 相机头文件
|       |-- config/             # 配置结构头文件
|       |-- detection/          # YOLO 检测头文件
|       |-- geometry/           # 标定/PnP/云台头文件
|       |-- tasks/              # 任务/CLI 头文件
|       |-- util/               # 文本工具头文件
|       `-- visualization/      # 绘制/HUD/跟踪头文件
`-- src/
    |-- main.cpp                # 所有任务共享的可执行程序入口
    `-- pnp_vision/
        |-- camera/             # OpenCV/海康相机实现
        |-- config/             # YAML 配置映射与校验
        |-- detection/          # YOLO 推理实现
        |-- geometry/           # 标定/PnP/云台实现
        |-- tasks/              # CLI、任务入口和工具函数
        `-- visualization/      # 绘制/HUD/跟踪实现
```

## 数据流

```text
相机取帧 -> YOLO 检测 -> PnP 解算 -> 云台角度 -> 绘制/HUD
                              ^
                              |
                     曝光/增益控制
```

实时任务采用双线程结构：

```text
采集线程               有界队列                处理线程
相机读帧 -> 写入队列 -> 容量 2，满时丢最旧 -> 弹出最新帧
                                                |
                                       检测/PnP/绘制/按键
```

队列缓存最多 2 帧。当推理速度低于相机帧率时，最旧的待处理帧会被丢弃，避免旧帧堆积造成持续延迟。

`calibrate_camera` 和 `adjust_exposure` 仍然使用单线程循环，便于逐帧交互。

## 构建

```bash
cmake -S . -B build
cmake --build build -j
```

构建要求：

- OpenCV：core、imgproc、calib3d、videoio、highgui、dnn
- yaml-cpp
- CMake 3.16+
- 可选 TensorRT/CUDA
- 可选 LibTorch
- 可选海康 MVS SDK

## 模型

运行前需要把 YOLO 模型放到 `model/` 目录，默认使用 `labubu_yolov8n.engine`。

如果使用 TensorRT，可先生成 engine 文件：

```bash
trtexec \
  --onnx=/home/zhaxiuyuan/桌面/PNPC++/model/labubu_yolov8n.onnx \
  --saveEngine=/home/zhaxiuyuan/桌面/PNPC++/model/labubu_yolov8n.engine
```

## 运行任务

```bash
# 1. 实时 YOLO 检测
./build/task1_yolo_detection

# 2. 检测 + PnP 距离
./build/task2_pnp_distance

# 3. 检测 + 位姿/欧拉角
./build/task3_pnp_pose

# 4. 完整可视化：检测 + PnP + HUD + 键盘曝光/增益
./build/task4_pnp_visualize

# 相机标定
./build/calibrate_camera

# 单独调曝光/增益
./build/adjust_exposure
```

程序启动时默认自动加载项目根目录下的 `config/config.yaml`，也可以手动指定：

```bash
./build/task1_yolo_detection --config my_camera.yaml --no-show
./build/task4_pnp_visualize --object-width 0.05 --object-height 0.05
```

常用命令行参数：

- `--camera laptop|hik`
- `--source <相机编号或视频路径>`
- `--model <模型路径>`
- `--runtime auto|tensorrt|libtorch|onnx`
- `--calib <标定文件路径>`
- `--conf <置信度>`
- `--nms <NMS 阈值>`
- `--imgsz <输入尺寸>`
- `--no-show`：不显示窗口
- `--fps-limit <帧率上限>`：`0` 表示不限制

task4 和 `adjust_exposure` 的曝光/增益按键：

- 方向键：粗调
- `-` / `=`：细调
- `q` 或 `Esc`：退出

## 配置文件

`config/config.yaml` 是主运行配置，包含：

- 相机模式和视频源
- 曝光/增益范围与步长
- 模型路径、运行后端、检测参数
- 目标物尺寸和 PnP 参数
- 跟踪参数
- 标定参数
- 窗口显示选项和帧率上限
