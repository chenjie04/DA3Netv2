from ultralytics import YOLO

model = YOLO("yolo114n.yaml")
# model = YOLO("runs/yolo114_VOC/n5/weights/best.pt")
# print(model)

model.info()