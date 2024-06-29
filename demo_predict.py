from ultralytics import YOLO

yolo = YOLO("runs/detect/Best-model/8n-C3_DualConv-bifpn3-ADown-CA-FocalMPDIou/weights/best.pt", task="detect")

results = yolo(source="datasets/bvn2/images/test/Orah_758.jpg", save=True)

