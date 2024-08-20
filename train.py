from ultralytics import YOLO
import torch

# 训练模型
# yolo detect train data=data/data.yaml model=models/yolov10s.pt epochs=600 batch=16 imgsz=640 device=0 amp=false
