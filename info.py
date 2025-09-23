from ultralytics import YOLO

model = YOLO("yolo114n.yaml")
# print(model)

model.info()