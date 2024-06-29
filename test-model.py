import contextlib
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from ultralytics import RTDETR, YOLO
from ultralytics.yolo.cfg import TASK2DATA
from ultralytics.yolo.data.build import load_inference_source
from ultralytics.yolo.utils import DEFAULT_CFG, LINUX, MACOS, ONLINE, ROOT, SETTINGS, WINDOWS
from ultralytics.yolo.utils.downloads import download
from ultralytics.yolo.utils.torch_utils import TORCH_1_9


CFG = ('ultralytics/models/v8/Atten-Contrast/ADown-C3_DualConv-BiFPN-SimAM_yolov8.yaml')


model = YOLO(CFG)
model(source=None, imgsz=32, augment=True)  # also test no source and augment

