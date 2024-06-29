from ultralytics import YOLO


# Load a model 加载预训练模型
# 添加注意力机制，SEAtt_yolov8.yaml 默认使用的是n

if __name__ == '__main__':
    from ultralytics import YOLO
    model = YOLO("ultralytics/models/v8/Atten-Contrast/ADown-C3_DualConv-BiFPN-SimAM_yolov8n.yaml")  # build a new model from scratch
#   model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="mydata.yaml", workers=8, epochs=300, batch=16, cache=True, resume=False, name='8n-ADown-C3_DualConv-BiFPN-SimAM')  # train the model



