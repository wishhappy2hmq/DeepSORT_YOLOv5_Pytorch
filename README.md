# DeepSORT + YOLOv5 目标跟踪项目

本项目结合了 YOLOv5 进行目标检测和 DeepSORT 进行多目标跟踪，提供了一个强大的实时跟踪解决方案。它旨在利用卷积神经网络和深度学习跟踪算法，实现视频中的多目标同时检测和跟踪。

## 功能特点
- **实时目标检测**：使用 YOLOv5 进行快速而准确的目标检测。
- **多目标跟踪**：集成了 DeepSORT，可以在多帧之间保持目标的身份。
- **可调设置**：可以轻松配置置信度阈值、输入视频和模型权重。

## 依赖安装
通过运行以下命令来安装所需的依赖项：
```sh
pip install -r requirements.txt
```

## 运行跟踪器
要运行目标跟踪器，使用以下命令：
```sh
python tracker.py --input_path ./data/video/demo.mp4 --weights yolov5/weights/best.pt
```
### 参数说明
- **`--input_path`**：输入视频文件的路径（例如：`./data/video/demo.mp4`）。
- **`--weights`**：YOLOv5 权重文件的路径。

## 项目结构
- **`tracker.py`**：运行目标跟踪的主脚本。
- **`yolov5/`**：包含 YOLOv5 模型代码。
- **`deep_sort_pytorch/`**：包含 DeepSORT 跟踪实现。
- **`data/`**：包含输入视频的文件夹。
- **`README.md`**：项目的设置和使用说明。
- **`requirements.txt`**：项目所需的依赖项列表。

## 安装步骤
1. 将此仓库克隆到本地：
   ```sh
   git clone https://github.com/yourusername/DeepSORT_YOLOv5_Pytorch.git
   cd DeepSORT_YOLOv5_Pytorch
   ```
2. 安装所需的依赖项：
   ```sh
   pip install -r requirements.txt
   ```

## 如何贡献
- **Fork** 此仓库。
- 创建一个新分支（`git checkout -b feature/YourFeature`）。
- 提交你的更改（`git commit -m 'Add a new feature'`）。
- 推送到新分支（`git push origin feature/YourFeature`）。
- 打开一个 **Pull Request**。

## 许可证
本项目基于 MIT 许可证。详情请参阅 `LICENSE` 文件。

## 鸣谢
- YOLOv5：用于先进的目标检测。
- DeepSORT：用于多目标跟踪。

