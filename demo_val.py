from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("runs/detect/train2/weights/best.pt")  # load a pretrained model (recommended for training)
#model.val(data="mydata.yaml", split='test', batch=8)
# Use the model
#model.train(data="coco128.yaml", epochs=3)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format
if __name__ == '__main__':
    model = YOLO("runs/detect/8n-ADown-C3_DualConv-BiFPN-SimAM/weights/best.pt")  # load a pretrained model (recommended for training)
    model.val(data="mydata.yaml", split='test', batch=1)
    # 代码
