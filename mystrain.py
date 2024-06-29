from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/v8/mango.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="ultralytics/cfg/datasets/mango.yaml", epochs=100)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
#results = model("IMG_20230722_112208.jpg")  # predict on an image
#path = model.export(format="onnx")  # export the model to ONNX format