import sys
import os
import torch
import cv2
import numpy as np

# 将 yolov5 和 deep_sort_pytorch 路径加入到系统路径中
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_sort_pytorch'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import non_max_suppression,scale_boxes, xyxy2xywh
from yolov5.utils.torch_utils import select_device
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from yolov5.utils.general import xyxy2xywh

class VideoTracker:
    def __init__(self, weights, config_deepsort, input_path, output_path, device='cpu', img_size=640):
        # 初始化设备
        self.device = select_device(device)

        # 初始化 YOLOv5
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False)
        self.model.eval()
        self.img_size = img_size

        # 初始化 DeepSORT
        cfg = get_config()
        cfg.merge_from_file(config_deepsort)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST,
                                 min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE,
                                 n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=device != 'cpu')

        # 初始化视频读取与保存
        self.input_path = input_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(input_path)
        assert self.cap.isOpened(), f"Failed to open video file {input_path}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, int(self.cap.get(cv2.CAP_PROP_FPS)),
                                   (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # 目标检测
            detections = self.detect(frame)

            # 目标追踪
            if detections is not None and len(detections):
                bbox_xywh, confs, cls_ids = self.prepare_for_deepsort(detections, frame)

                # 如果 bbox_xywh 为空，跳过这一帧
                if len(bbox_xywh) == 0:
                    print("No valid detections in this frame, skipping tracking...")
                    self.out.write(frame)
                    continue

                # 打印调试信息
                print("bbox_xywh shape:", bbox_xywh.shape)  # 应该是 (N, 4)
                print("confs shape:", confs.shape)  # 应该是 (N, 1)
                print("cls_ids shape:", cls_ids.shape)  # 应该是 (N, 1)
                print("frame shape:", frame.shape)  # 应该是 (height, width, channels)

                # 传递 bbox_xywh, confs, frame, cls_ids 到 update 方法中
                outputs = self.deepsort.update(bbox_xywh, confs, cls_ids, frame)
                self.draw_boxes(frame, outputs)
            else:
                print("No detections found in this frame.")

            # 保存结果
            self.out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def detect(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).float().to(self.device).permute(2, 0, 1).unsqueeze(0)
        img /= 255.0

        with torch.no_grad():
            pred = self.model(img)

        # 非极大抑制
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)
        return pred[0]

    def prepare_for_deepsort(self, detections, frame):
        # 如果 detections 为空，返回空张量
        if detections is None or len(detections) == 0:
            return torch.zeros((0, 4)), torch.zeros((0, 1)), torch.zeros((0, 1))

        # 使用 YOLOv5 提供的 scale_boxes 函数还原检测框至原始图像尺寸
        img1_shape = (self.img_size, self.img_size)  # 输入图像的尺寸 (height, width)
        img0_shape = frame.shape[:2]  # 原始图像的尺寸 (height, width)
        detections[:, :4] = scale_boxes(img1_shape, detections[:, :4], img0_shape).round()

        # 转换为中心点坐标和宽高格式
        bbox_xywh = xyxy2xywh(detections[:, :4])
        confs = detections[:, 4:5]  # 置信度，确保形状为 (N, 1)
        cls_ids = detections[:, 5:6]  # 类别标签，确保形状为 (N, 1)

        return bbox_xywh.cpu(), confs.cpu(), cls_ids.cpu()

    def draw_boxes(self, frame, outputs):
        for output in outputs:
            # 修改为解包 6 个值
            x1, y1, x2, y2, track_id, class_id = output

            # 在图像上绘制边界框和跟踪 ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'ID: {track_id}, Class: {class_id}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


if __name__ == '__main__':
    # 参数设置
    weights = 'yolov5/weights/yolov5s.pt'  # YOLOv5 权重文件
    config_deepsort = 'deep_sort_pytorch/configs/deep_sort.yaml'  # DeepSORT 配置文件
    input_path = 'input.mp4'  # 输入视频文件
    output_path = 'output_video.mp4'  # 输出视频文件
    device = 'cpu'  # 使用 GPU (指定为 'cpu' 则使用 CPU)

    tracker = VideoTracker(weights, config_deepsort, input_path, output_path, device)
    tracker.run()
