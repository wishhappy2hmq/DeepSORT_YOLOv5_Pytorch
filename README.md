# DeepSORT + YOLOv5 目标跟踪项目

项目结合了最新版YOLOv5 进行目标检测和 DeepSORT 进行多目标跟踪，逻辑清晰

## 功能特点
yolov5+deepsort互相独立工作，推理流程非常清晰

## 依赖安装
通过运行以下命令来安装所需的依赖项：
```sh
pip install -r requirements.txt
```

## 运行
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


